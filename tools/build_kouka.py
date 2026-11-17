# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""English school-anthem sheets (Extras > Anthem, layout option A).

Reads tl/KOUKA.txt (## anthem N rec M, RO = romaji, EN = English).
Writes private/build/KOUKA.BIN (6-word draw patch + rewritten records) and
private/build/KOUKA00-04.BIZ (+ .BIZ.lz) with a new lyric sheet.
--png DIR writes 4:3 previews (DIR/anthemN.png) on the darkened
background. --check builds everything but writes nothing to private/build/.

Display (private/re/anthem/notes.org): each record draws one SPRT from the
sheet TIM (4bpp, VRAM 768,0, CLUT 0,480) with additive blending. After the
patch a record is (start frame, no-fade, tpage x/64, x, y, u + v<<8) and
the sprite is 256x32. The sheet is 1024x256 (tpages 12-15), 8 strips of
256x32 per tpage; record i uses strip i. Strips hold RO on top (16 px) and
EN below (13 px). Text is drawn at the 4:3 display aspect (x squashed by
0.8, one VRAM pixel shows 1.25 wide).
"""
import sys, os, re, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import lz

TL = 'tl/KOUKA.txt'
KBIN = 'private/disc/DAT1/KOUKA.BIN'
BASE = 0x8009c000
TTF = '/usr/share/fonts/TTF/'
SS = 5                         # supersampling
SHEET = 0x3c800                # sheet TIM offset in KOUKA0n.BIZ
PATCH = [(0x8009c994, 0x01281023, 0x00000000),    # x = rec[3]
         (0x8009c9a4, 0x00021100, 0x00000000),    # y = rec[4]
         (0x8009c9a8, 0x24420080, 0x00000000),
         (0x8009c9b4, 0x24020010, 0x24020100),    # w = 256
         (0x8009c9bc, 0x24020100, 0x24020020),    # h = 32
         (0x8009c9c8, 0xa2030014, 0xa6030014)]    # sh: u + v<<8

# style: font file, px, gain (brightness 0..1), tracking px (display)
ST = {'ro':   ('pala.ttf', 16, 1.00, 0.0),
      'en':   ('palai.ttf', 13, 0.92, 0.0),
      'tro':  ('palai.ttf', 13, 0.85, 0.4),
      'ten':  ('palab.ttf', 18, 1.00, 0.3),
      'cred': ('palai.ttf', 13, 0.80, 0.0),
      'mark': ('palabi.ttf', 13, 0.80, 0.6)}
MAXW = 232                     # widest text in a strip (overscan)
RO_BASE, EN_BASE = 13, 27      # baselines in a lyric strip
T_BASE, TE_BASE = 10, 27       # baselines in the title strip
GAP = 28                       # px between the two columns

# Per anthem: free x range, header top, column top, keep-out boxes
# (painted school names, x0, y0, x1, y1).
LAY = {0: dict(L=60, R=492, top=18, col=88, pitch=34, keep=[(0, 16, 52, 336)]),
       1: dict(L=20, R=492, top=18, col=88, pitch=32, keep=[(8, 8, 124, 48)]),
       2: dict(L=20, R=492, top=8, col=76, pitch=31, hr=494,
               keep=[(0, 0, 340, 72), (190, 418, 512, 474)]),
       3: dict(L=20, R=492, top=18, col=136, pitch=31,
               keep=[(0, 16, 64, 126), (96, 24, 190, 152), (310, 350, 512, 480)]),
       4: dict(L=20, R=492, top=18, col=88, pitch=34, keep=[(340, 416, 508, 474)])}

def load_tl(path=TL):
    t = open(path, encoding='utf-8').read()
    out = {}
    for a, r, ro, en in re.findall(r'## anthem (\d) rec (\d+)\nJP: .*\nRO: ?(.*)\nEN: ?(.*)', t):
        out.setdefault(int(a), {})[int(r)] = (ro.strip(), en.strip())
    return out

_fc = {}
def font(path, px):
    k = (path, px)
    if k not in _fc: _fc[k] = ImageFont.truetype(TTF + path, px * SS)
    return _fc[k]

def line(text, style, px=None, track=None):
    """Coverage (0..1) of one line at VRAM resolution, and its baseline row."""
    path, px0, gain, tr0 = ST[style]
    px = px or px0; track = tr0 if track is None else track
    f = font(path, px)
    asc, desc = f.getmetrics()
    yo = (-asc) % SS
    w = f.getlength(text) + track * SS * len(text) + 4 * SS
    W = int(np.ceil(w / SS * 0.8)); H = int(np.ceil((yo + asc + desc) / SS))
    im = Image.new('L', (int(np.ceil(W * SS / 0.8)), H * SS), 0)
    d = ImageDraw.Draw(im)
    for i, ch in enumerate(text):
        d.text((f.getlength(text[:i]) + track * SS * i + 2 * SS, yo), ch, font=f, fill=255)
    im = im.resize((W, H), Image.BOX, box=(0, 0, W * SS / 0.8, H * SS))
    a = np.array(im) / 255.0
    cols = np.nonzero(a.max(0) > 0.08)[0]
    a = a[:, :cols.max() + 1] if len(cols) else a[:, :0]   # keep the pen origin
    return a * gain, (yo + asc) // SS

def fit(text, style, notes):
    """Render a line no wider than MAXW: tighten tracking, then shrink."""
    path, px, gain, tr = ST[style]
    for p in (px, px - 1):
        for t in (tr, tr - 0.3, tr - 0.6):
            a, b = line(text, style, p, t)
            if a.shape[1] <= MAXW:
                if (p, t) != (px, tr): notes.append(f'condensed {p}px track {t:+.1f}: {text}')
                return a, b
    notes.append(f'TOO WIDE ({a.shape[1]}): {text}')
    return a, b

def put(out, a, base, row, u):
    """Max-composite coverage a with its baseline at row, left edge at u."""
    y = row - base
    h, w = a.shape
    s0 = max(0, -y); s1 = min(h, 32 - y)
    out[y + s0:y + s1, u:u + w] = np.maximum(out[y + s0:y + s1, u:u + w], a[s0:s1])

def strips(tl, notes):
    """Return {rec: (32x256 coverage, kind)} for one anthem."""
    S = {}
    for r, (ro, en) in tl.items():
        out = np.zeros((32, 256))
        if r == 0:
            a, b = fit(en, 'ten', notes); ae, be = fit(ro, 'tro', notes)
            # right-aligned title block
            put(out, ae, be, T_BASE, 252 - ae.shape[1])
            put(out, a, b, TE_BASE, 252 - a.shape[1]); kind = 'title'
        elif r == 1:
            a, b = fit(en, 'cred', notes); put(out, a, b, 12, 252 - a.shape[1]); kind = 'cred'
        elif not ro:
            a, b = fit(en, 'mark', notes); put(out, a, b, 12, 0); kind = 'mark'
        else:
            a, b = fit(ro, 'ro', notes); put(out, a, b, RO_BASE, 0)
            a, b = fit(en, 'en', notes); put(out, a, b, EN_BASE, 0); kind = 'line'
        S[r] = (out, kind)
    return S

def ink(c):
    """Ink box of a strip: (u0, row0, u1, row1), exclusive ends."""
    ys = np.nonzero(c.max(1) > 0.05)[0]; xs = np.nonzero(c.max(0) > 0.05)[0]
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

def layout(n, S):
    """Screen (x, y) for each record's strip."""
    cfg = LAY[n]
    recs = sorted(S)
    m2 = [r for r in recs if r > 2 and S[r][1] == 'mark'][0]
    v1 = [r for r in recs if 2 < r < m2]; v2 = [r for r in recs if r > m2]
    w1 = max(ink(S[r][0])[2] for r in v1 + [2])
    w2 = max(ink(S[r][0])[2] for r in v2 + [m2])
    block = w1 + GAP + w2
    x1 = cfg['L'] + (cfg['R'] - cfg['L'] - block) // 2
    x2 = x1 + w1 + GAP
    right = cfg.get('hr', x2 + w2)         # header right edge
    pos = {}
    pos[0] = (right - 252, cfg['top'])
    t1 = ink(S[0][0])[3]; c0 = ink(S[1][0])[1]
    pos[1] = (right - 252, cfg['top'] + t1 + 5 - c0)
    lt = min(ink(S[r][0])[1] for r in v1 + v2)
    pitch = cfg['pitch']
    mb = ink(S[2][0])[3]
    for xm, m, v in ((x1, 2, v1), (x2, m2, v2)):
        pos[m] = (xm, cfg['col'])
        y = cfg['col'] + mb + 6 - lt
        for r in v:
            pos[r] = (xm, y); y += pitch
    return pos, pitch

def check(n, S, pos):
    """Keep-out and screen checks; returns a list of problems."""
    bad = []
    for r, (x, y) in pos.items():
        u0, r0, u1, r1 = ink(S[r][0])
        X0, Y0, X1, Y1 = x + u0, y + r0, x + u1, y + r1
        if X0 < 16 or X1 > 496 or Y0 < 8 or Y1 > 466:
            bad.append(f'rec {r} outside safe area ({X0},{Y0})-({X1},{Y1})')
        for k in LAY[n]['keep']:
            if X0 < k[2] and k[0] < X1 and Y0 < k[3] and k[1] < Y1:
                bad.append(f'rec {r} hits painted name {k}')
    return bad

# CLUT: 0 white, 1-14 grey ramp, 15 transparent (all with STP, as the JP)
LEV = [31] + [round(31 * i / 15) for i in range(1, 15)]
CLUT = [(v | v << 5 | v << 10 | 0x8000) for v in LEV] + [0]

def quant(c):
    """Coverage -> CLUT index."""
    lv = np.array(LEV + [0], float) / 31
    idx = np.abs(c[..., None] - lv[None, None, :]).argmin(-1)
    return idx.astype(np.uint8)

BURST = (0xe00, 0x1000, 0x1200, 0x1400)   # originals reach 0x14e8

def bgmerge(d):
    """Merge background CLUT entries that show the same colour at the
    draw brightness 0x30 (5-bit channel * 0x30 >> 7), so the LZ finds more
    repeats. The bg TIM is 8bpp 512x480 at offset 0."""
    bl = struct.unpack_from('<I', d, 8)[0]
    clut = struct.unpack_from('<256H', d, 20)
    q = 8 + bl
    blen, x, y, w, h = struct.unpack_from('<IHHHH', d, q)
    first, remap = {}, np.zeros(256, np.uint8)
    for i, v in enumerate(clut):
        k = tuple(((v >> s & 31) * 0x30) >> 7 for s in (0, 5, 10)) + (v >> 15,)
        remap[i] = first.setdefault(k, i)
    a = q + 12; b = a + w * 2 * h
    d[a:b] = remap[np.frombuffer(bytes(d[a:b]), np.uint8)].tobytes()

def tim(sheet):
    px = (sheet[:, 0::2] | (sheet[:, 1::2] << 4)).astype(np.uint8).tobytes()
    h = struct.pack('<II', 0x10, 8)
    h += struct.pack('<IHHHH', 12 + 32, 0, 480, 16, 1) + struct.pack('<16H', *CLUT)
    h += struct.pack('<IHHHH', 12 + len(px), 768, 0, 256, sheet.shape[0])
    return h + px

def build(write=True, png=None):
    tl = load_tl()
    k = bytearray(open(KBIN, 'rb').read())
    for ram, old, new in PATCH:
        o = ram - BASE
        assert struct.unpack_from('<I', k, o)[0] == old, hex(ram)
        struct.pack_into('<I', k, o, new)
    problems = 0
    out = {}
    for n in range(5):
        notes = []
        cnt = struct.unpack_from('<h', k, 0x4d8 + 6 * n)[0] + 1
        p = struct.unpack_from('<I', k, 0x4f8 + 4 * n)[0] - BASE
        assert sorted(tl[n]) == list(range(cnt)), (n, cnt)
        S = strips(tl[n], notes)
        pos, pitch = layout(n, S)
        bad = check(n, S, pos)
        rows = 32 * -(-cnt // 4)
        sheet = np.full((rows, 1024), 15, np.uint8)
        cov = np.zeros((rows, 1024))
        for r in range(cnt):
            tx, v = 12 + r % 4, 32 * (r // 4)
            q = quant(S[r][0])
            sheet[v:v + 32, (tx - 12) * 256:(tx - 11) * 256] = q
            cov[v:v + 32, (tx - 12) * 256:(tx - 11) * 256] = np.array(LEV + [0])[q] / 31
            t, nf = struct.unpack_from('<2h', k, p + 12 * r)
            struct.pack_into('<5hH', k, p + 12 * r, t, nf, tx, pos[r][0], pos[r][1], v << 8)
        name = f'KOUKA0{n}'
        d = bytearray(open(f'private/unpacked/{name}.BIZ.lz', 'rb').read())
        bl = struct.unpack_from('<I', d, SHEET + 8)[0]
        end = SHEET + 8 + bl + struct.unpack_from('<I', d, SHEET + 8 + bl)[0]
        assert not any(d[end:]), name
        bgmerge(d)
        d = d[:SHEET] + tim(sheet)
        d += bytes(-len(d) % 0x800)
        slot = os.path.getsize(f'private/disc/DAT2/{name}.BIZ'); slot += -slot % 2048
        for cap in BURST:
            c = lz.compress(bytes(d), burst=cap)
            if len(c) <= slot: break
        if cap != BURST[0]: notes.append(f'burst cap {cap:#x}')
        ok = len(c) <= slot
        if not ok: bad.append(f'{len(c)} bytes > slot {slot}')
        problems += len(bad)
        print(f'{name}: {cnt} strips, pitch {pitch}, {len(c)}/{slot} bytes'
              + ''.join(f'\n  {s}' for s in notes + bad))
        out[name] = (d, c, ok)
        if png: preview(n, cov, pos, cnt, f'{png}/anthem{n}.png')
    if write:
        for name, (d, c, ok) in out.items():
            if not ok: continue
            open(f'private/build/{name}.BIZ.lz', 'wb').write(d)
            open(f'private/build/{name}.BIZ', 'wb').write(c)
        open('private/build/KOUKA.BIN', 'wb').write(k)
    print(f'KOUKA: problems {problems}')
    return problems

def preview(n, cov, pos, cnt, path):
    sys.path.insert(0, 'private/re/inventory')
    from render import image
    bg, _, _ = image(f'private/unpacked/KOUKA0{n}.BIZ.lz', 0)
    bg = np.array(bg.convert('RGB')).astype(float) * 0x30 / 0x80
    for r in range(cnt):
        tx, v = r % 4, 32 * (r // 4)
        s = cov[v:v + 32, tx * 256:tx * 256 + 256] * 255 * 0x70 / 0x80
        x, y = pos[r]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(512, x + 256), min(480, y + 32)
        bg[y0:y1, x0:x1] += s[y0 - y:y1 - y, x0 - x:x1 - x, None]
    im = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    im.resize((640, 480), Image.LANCZOS).save(path)

if __name__ == '__main__':
    a = sys.argv[1:]
    png = a[a.index('--png') + 1] if '--png' in a else None
    if png: os.makedirs(png, exist_ok=True)
    sys.exit(1 if build(write='--check' not in a, png=png) else 0)
