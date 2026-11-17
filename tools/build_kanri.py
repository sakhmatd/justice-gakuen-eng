# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""EDIT Register screens in English: KANRI.BIN glyph strings and the
EDITMOJI.TIZ header and school labels.

Reads tl/KANRI.txt (ID: EN), data/latin.tsv (MOJI2 Latin
codes), private/unpacked/MOJI2.TIZ.lz (glyph shapes, for centring) and
private/unpacked/EDITMOJI.TIZ.lz. Edits private/build/KANRI.BIN (written by
build_editcm.py; the disc copy if it is missing) and writes
private/build/EDITMOJI.TIZ. Run it after build_editcm.py. build_editcm.py also
calls patch_kanri() so it never drops the strings. --check only reports
problems; --png writes the redrawn art to private/re/register/gfx/.

KANRI facts (topic register): string records at 0x434 + 12k (s16 x, s16 y,
u32 RAM pointer, s16 tighten, u16 0), values at 0x5cc (6 records). The
drawer gives every cell 0x38 bytes of primitive buffer: buttons 6 cells,
status labels 8, panel strings and values 16. Strings are u16 count + u16
codes; they go to the old pools, then to the end of the file (KANRI can
grow to 0x7000). The pitch is 20 - tighten.
"""
import sys, os, struct, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import lz, burst
from build_sysmsg import charmap

KANRI_DISC = 'private/disc/DAT1/KANRI.BIN'
KANRI_OUT = 'private/build/KANRI.BIN'
EM_SRC = 'private/unpacked/EDITMOJI.TIZ.lz'
EM_DISC = 'private/disc/DAT2/EDITMOJI.TIZ'
EM_OUT = 'private/build/EDITMOJI.TIZ'
MOJI = 'private/unpacked/MOJI2.TIZ.lz'
TL = 'tl/KANRI.txt'
BASE = 0x8009c000
ORIG_LEN, MAX_LEN = 0x69c4, 0x7000
POOLS = [(0x258, 0x434), (0x59c, 0x5cc)]
REC, VREC = 0x434, 0x5cc
BURST = 0x1600
IMPACT = '/usr/share/fonts/TTF/impact.ttf'
ARIALBD = '/usr/share/fonts/TTF/arialbd.ttf'

# id: (record offset, cell limit, centre x or None, tighten)
BTN_C, PANEL_C = 431, 202
LAYOUT = {}
for k in range(6): LAYOUT[f's{k}'] = (REC + 12 * k, 6, BTN_C, 6)
for k in (7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 28, 29):
    LAYOUT[f's{k}'] = (REC + 12 * k, 16, PANEL_C, 7)
LAYOUT['s9'] = (REC + 12 * 9, 16, 242, 7)
LAYOUT['s8'] = (REC + 12 * 8, 12, 240, 6)
for k in range(19, 24): LAYOUT[f's{k}'] = (REC + 12 * k, 8, 94, 6)
for k in (24, 25): LAYOUT[f's{k}'] = (REC + 12 * k, 8, 422, 6)
for k in range(5): LAYOUT[f'v{k}'] = (VREC + 12 * k, 16, 94, 6)
KEEP = ('s6', 's26', 's27')
LAYOUT_KEEP = {'s6': REC + 12 * 6, 's26': REC + 12 * 26, 's27': REC + 12 * 27}
# Yes/No: one string (record 18), 64x20 cursor boxes at x 120 and 208
YESNO_REC, YESNO_BOX = REC + 12 * 18, (152, 240)
# status "N本": drop 本 (stack word 0xa9 -> terminator 0x3001), centre the digit
NHON_GLYPH, NHON_X = 0x5068, 0x5054

# EDITMOJI: header at texels 512-1023 x 120-167 (two 256-wide sprites),
# school labels in the free block x 512-1023 y 214-255
HDR_REC = (0x196, 0x1ac)
HDR_Y, HDR_H, HDR_CX = 120, 48, 257
SCH_REC = 0x754
SCH_SLOTS = [(512, 214), (600, 214), (512, 234), (600, 234)]
SCH_W, SCH_H = 88, 20
EM_PIX = 0x820

def load_tl(path=TL):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

# ---------------------------------------------------------------- glyphs
def moji_glyphs():
    d = lz.decompress(open(MOJI, 'rb').read())[0] if MOJI.endswith('.TIZ') else open(MOJI, 'rb').read()
    p = 8 + struct.unpack_from('<I', d, 8)[0] + 12
    raw = np.frombuffer(d[p:p + 512 * 256], np.uint8).reshape(256, 512)
    px = np.zeros((256, 1024), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return px

def glyph(px, c):
    t, r, col = c >> 8, c >> 4 & 15, c & 15
    return px[20 * r:20 * r + 20, 256 * t + 20 * col:256 * t + 20 * col + 20]

def ink(px, codes, tighten):
    """(left, right) ink columns of a string relative to its x."""
    lo, hi = None, None
    for i, c in enumerate(codes):
        if c == 0xfff: continue
        cols = np.where(glyph(px, c).any(0))[0]
        if not len(cols): continue
        a = i * (20 - tighten)
        lo = a + cols[0] if lo is None else min(lo, a + cols[0])
        hi = a + cols[-1] if hi is None else max(hi, a + cols[-1])
    return lo, hi

def encode(text, cm):
    out = []
    for ch in text:
        if ch not in cm: raise ValueError(f'no glyph for {ch!r} in {text!r}')
        out.append(cm[ch])
    return out

def yesno(px, cm, yes, no):
    """One string 'Yes' + blanks + 'No' whose words centre in the boxes."""
    best = None
    for t in range(4, 10):
        a = 20 - t
        cy, cn = encode(yes, cm), encode(no, cm)
        for gap in range(1, 10):
            codes = cy + [0xfff] * gap + cn
            if len(codes) > 16: continue
            l1, r1 = ink(px, cy, t)
            l2, r2 = ink(px, cn, t)
            off = (len(cy) + gap) * a
            # x such that both words centre; error is the mismatch
            x1 = YESNO_BOX[0] - (l1 + r1 + 1) / 2
            x2 = YESNO_BOX[1] - (off + (l2 + r2 + 1) / 2)
            x = round((x1 + x2) / 2); err = abs(x1 - x2) + 0.3 * abs(t - 6)
            if best is None or err < best[0]: best = (err, codes, x, t)
    return best

# ---------------------------------------------------------------- KANRI
def patch_kanri(k, tl=None, problems=None):
    """Apply the strings to a KANRI image (bytearray, Team-Up records from
    build_editcm kept). Returns the patched bytearray."""
    tl = load_tl() if tl is None else tl
    problems = [] if problems is None else problems
    disc = open(KANRI_DISC, 'rb').read()
    k = bytearray(k[:ORIG_LEN])
    for a, b in POOLS: k[a:b] = disc[a:b]
    k[REC:REC + 30 * 12] = disc[REC:REC + 30 * 12]
    k[VREC:VREC + 6 * 12] = disc[VREC:VREC + 6 * 12]
    cm = charmap(); gpx = moji_glyphs()
    strings = []                                  # (record offsets, codes, x, tighten)
    for key, (rec, lim, cx, t) in LAYOUT.items():
        en = tl.get(key, '')
        if key in KEEP or en in ('', '='): continue
        try: codes = encode(en, cm)
        except ValueError as e: problems.append(f'{key}: {e}'); continue
        if len(codes) > lim: problems.append(f'{key}: {en!r} has {len(codes)} cells > {lim}'); continue
        lo, hi = ink(gpx, codes, t)
        x = round(cx - (lo + hi + 1) / 2)
        recs = [rec] + ([VREC + 12 * 5] if key == 'v4' else [])
        strings.append((recs, codes, x, t, key))
    if tl.get('s18.yes') and tl.get('s18.no'):
        err, codes, x, t = yesno(gpx, cm, tl['s18.yes'], tl['s18.no'])
        strings.append(([YESNO_REC], codes, x, t, 's18'))
    for key in tl:
        if key not in LAYOUT and key not in KEEP + ('s18.yes', 's18.no', 'hdr') and not key.startswith('sch'):
            problems.append(f'{key}: unknown id')
    # place: old pools first, then the end of the file. The kept Latin
    # strings (NO DATA, CLEAR, DATA ERROR) stay where they are: code
    # loads their addresses directly.
    keep = []
    for key in KEEP:
        rec = LAYOUT_KEEP[key]
        o = struct.unpack_from('<I', disc, rec + 4)[0] - BASE
        n = struct.unpack_from('<H', disc, o)[0]
        keep.append((o, o + ((2 + 2 * n + 3) & ~3)))
    free = []
    for a, b in POOLS:
        for ka, kb in sorted(keep):
            if a <= ka < b:
                if ka > a: free.append([a, ka])
                a = kb
        if b > a: free.append([a, b])
    tail = []
    for recs, codes, x, t, key in strings:
        blob = struct.pack('<H%dH' % len(codes), len(codes), *codes)
        blob += b'\0' * (-len(blob) % 4)
        for f in free:
            if f[1] - f[0] >= len(blob):
                at = f[0]; k[at:at + len(blob)] = blob; f[0] += len(blob); break
        else:
            at = len(k); k += blob; tail.append(key)
        for r in recs:
            y = struct.unpack_from('<h', k, r + 2)[0]
            struct.pack_into('<hhIhH', k, r, x, y, BASE + at, t, 0)
    for f in free: k[f[0]:f[1]] = b'\0' * (f[1] - f[0])
    # name code table (s16 at 0x934): codes 61/62 point at MOJI2 0x4c/0x4d,
    # not at y/z (0x050/0x051)
    struct.pack_into('<2h', k, 0x934 + 2 * 61, cm['y'], cm['z'])
    # Latin names: unpack letter pairs in the name drawer
    k[NAME_SITE:NAME_SITE + 4] = disc[NAME_SITE:NAME_SITE + 4]
    assert struct.unpack_from('<I', k, NAME_SITE)[0] == 0xafb3001c
    k += b'\0' * (-len(k) % 4)
    org = BASE + len(k)
    code = unpack_code(org)
    k += code
    struct.pack_into('<I', k, NAME_SITE, 0x0c000000 | (org >> 2 & 0x3ffffff))
    if len(k) > MAX_LEN: problems.append(f'KANRI {len(k):#x} > {MAX_LEN:#x}')
    # N本 -> N
    struct.pack_into('<I', k, NHON_GLYPH, 0x24020000 | 0x3001)
    struct.pack_into('<I', k, NHON_X, 0x24070000 | 84)
    return k

NAME_SITE = 0x198c          # 0x8009d98c sw s3,0x1c(sp) in the name drawer

def unpack_code(org):
    """Name drawer 0x8009d95c hook (jal at 0x8009d98c, the original
    sw s2 stays in the delay slot). Rewrites the code string at s6 in
    place: letter pairs 0x8000 | c1 << 7 | c2 (our name entry) become
    single table codes (1-9, 0 = 10, A-Z 11-36, a-z 37-62, others 0x61e;
    blanks and 0x7f dropped), at most 4 glyphs (the callers' primitive
    buffers hold 4). JP strings pass through unchanged. Uses a 16-byte
    scratch copy after the code. Only temporaries are used; a0-a3 are
    already saved in s5, s6, fp, s4."""
    from hook import Asm
    a = Asm(org)
    a.sw('s3', 0x1c, 'sp')                       # the replaced instruction
    a.addiu('t3', 'zero', 0x3001)
    scratch = org + 4 * 80
    a.lui('t0', (scratch + 0x8000) >> 16); a.addiu('t0', 't0', scratch & 0xffff)
    a.sh('t3', 12, 't0')                         # guard after 6 copied words
    a.addu('t1', 's6', 'zero'); a.addiu('t2', 'zero', 6)
    a.label('copy')
    a.lhu('v0', 0, 't1'); a.addiu('t1', 't1', 2)
    a.sh('v0', 0, 't0'); a.addiu('t0', 't0', 2)
    a.branch('beq', 'v0', 't3', 'copied'); a.addiu('t2', 't2', -1)
    a.branch('bne', 't2', 'zero', 'copy'); a.nop()
    a.label('copied')
    a.lui('t0', (scratch + 0x8000) >> 16); a.addiu('t0', 't0', scratch & 0xffff)
    a.addu('t1', 's6', 'zero'); a.addu('t2', 'zero', 'zero')
    a.label('loop')
    a.lhu('v0', 0, 't0'); a.addiu('t0', 't0', 2)
    a.branch('beq', 'v0', 't3', 'end'); a.andi('v1', 'v0', 0x8000)
    a.branch('bne', 'v1', 'zero', 'pair'); a.sltiu('at', 't2', 5)
    a.branch('beq', 'at', 'zero', 'end'); a.nop()
    a.sh('v0', 0, 't1'); a.addiu('t1', 't1', 2)
    a.branch('beq', 'zero', 'zero', 'loop'); a.addiu('t2', 't2', 1)
    a.label('pair')
    a.srl('a2', 'v0', 7); a.andi('a2', 'a2', 0x7f)
    a.andi('a1', 'v0', 0x7f); a.addiu('t4', 'zero', 2)
    a.label('emit')
    a.sltiu('at', 't2', 4)
    a.branch('beq', 'at', 'zero', 'end'); a.addiu('v1', 'a2', -0x61)
    a.sltiu('at', 'v1', 26)
    a.branch('bne', 'at', 'zero', 'store'); a.addiu('v1', 'v1', 37)
    a.addiu('v1', 'a2', -0x41); a.sltiu('at', 'v1', 26)
    a.branch('bne', 'at', 'zero', 'store'); a.addiu('v1', 'v1', 11)
    a.addiu('v1', 'a2', -0x31); a.sltiu('at', 'v1', 9)
    a.branch('bne', 'at', 'zero', 'store'); a.addiu('v1', 'v1', 1)
    a.addiu('at', 'a2', -0x30)
    a.branch('beq', 'at', 'zero', 'store'); a.addiu('v1', 'zero', 10)
    a.addiu('at', 'a2', -0x20)
    a.branch('beq', 'at', 'zero', 'next'); a.addiu('at', 'a2', -0x7f)
    a.branch('beq', 'at', 'zero', 'next'); a.nop()
    a.branch('beq', 'a2', 'zero', 'next'); a.addiu('v1', 'zero', 0x61e)
    a.label('store')
    a.sh('v1', 0, 't1'); a.addiu('t1', 't1', 2); a.addiu('t2', 't2', 1)
    a.label('next')
    a.addiu('t4', 't4', -1)
    a.branch('bne', 't4', 'zero', 'emit'); a.addu('a2', 'a1', 'zero')
    a.branch('beq', 'zero', 'zero', 'loop'); a.nop()
    a.label('end')
    a.sh('t3', 0, 't1')
    a.jr('ra'); a.nop()
    assert len(a.code) <= 80, len(a.code)
    while len(a.code) < 80: a.nop()
    for _ in range(4): a.emit(0)                 # scratch, 8 halfwords
    return a.finish()

# ---------------------------------------------------------------- art
def em_load():
    d = bytearray(open(EM_SRC, 'rb').read())
    raw = np.frombuffer(bytes(d[EM_PIX:EM_PIX + 512 * 256]), np.uint8).reshape(256, 512)
    px = np.zeros((256, 1024), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return d, px

def em_store(d, px):
    d[EM_PIX:EM_PIX + 512 * 256] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()

def em_clut(d, cx, cy=481):
    return list(struct.unpack_from('<16H', d, 0x14 + ((cy - 481) * 512 + cx) * 2))

def text_mask(parts, ss=4, shear=0.0, track=0, font=IMPACT):
    """parts: [(text, cap px)] on one baseline. Returns a float mask at 1x
    (rows = cap of the largest part + margins), baseline row, ink box."""
    capmax = max(c for _, c in parts)
    H = int(capmax * 2.0) * ss; W = 1200 * ss
    im = Image.new('L', (W, H), 0); dr = ImageDraw.Draw(im)
    x = 40 * ss; base = int(capmax * 1.4) * ss
    for text, cap in parts:
        size = 100
        f = ImageFont.truetype(font, size)
        ch = f.getbbox('H'); size = round(size * cap / (ch[3] - ch[1]))
        f = ImageFont.truetype(font, size * ss)
        top = f.getbbox('H')[1]
        asc = f.getmetrics()[0]
        for c in text:
            dr.text((x, base - (f.getbbox('H')[3])), c, font=f, fill=255)
            x += f.getlength(c) + track * ss
    a = np.asarray(im, np.float32) / 255
    if shear:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).transform(
            im.size, Image.AFFINE, (1, shear, -shear * base, 0, 1, 0), Image.BICUBIC), np.float32) / 255
    a = a.reshape(H // ss, ss, W // ss, ss).mean((1, 3))
    return a, base // ss

def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + r * 0.6

def shift(m, dx, dy):
    o = np.zeros_like(m)
    H, W = m.shape
    o[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = m[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return o

def header_art(parts, out_r=4, sh=(2, 4), shear=0.22, track=1):
    """Index image (rows HDR_H) in the G1 palette: yellow face with a
    vertical gradient, thin bevel, red outline, dark rim, orange
    extrusion and a maroon drop shadow."""
    a, base = text_mask(parts, shear=shear, track=track)
    face = a >= 0.5
    ext = (shift(face, 1, 1) | shift(face, 2, 2)) & ~face          # orange extrusion
    body = face | ext
    out = ndimage.binary_dilation(body, disk(out_r)) & ~body       # red outline
    rim = ndimage.binary_dilation(body | out, disk(1)) & ~(body | out)
    allm = body | out | rim
    shadow = shift(ndimage.binary_dilation(allm, disk(0)), *sh) & ~allm
    srim = ndimage.binary_dilation(shadow | allm, disk(1)) & ~(shadow | allm)
    srim &= shift(srim | shadow, 0, 0)
    idx = np.zeros(a.shape, np.uint8)
    idx[srim & ~ndimage.binary_dilation(allm, disk(1))] = 0
    idx[shadow] = 13
    idx[rim] = 12
    idx[out] = 10
    idx[ext] = 8
    # face gradient over the cap height of the face rows
    rows = np.where(face.any(1))[0]; t0, t1 = rows[0], rows[-1]
    yy = (np.arange(a.shape[0]) - t0) / max(1, t1 - t0)
    grad = np.select([yy < 0.5, yy < 0.78], [2, 3], 4).astype(np.uint8)
    fidx = np.repeat(grad[:, None], a.shape[1], 1)
    idx[face] = fidx[face]
    # bevel: face pixels whose lower-right neighbour is not face
    bev = face & ~shift(face, -1, -1)
    idx[bev & (yy[:, None] > 0.5)] = 1
    ys, xs = np.where(idx > 0)
    return idx[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def school_art(text, cap=11, track=0, font=ARIALBD, thr=0.4):
    """Index image (<= 88 x 20) in the G2 palette: horizontal deep-pink to
    light-pink gradient with a dark-red drop shadow, like the JP labels."""
    a, base = text_mask([(text, cap)], track=track, font=font)
    face = a >= thr
    edge = (a >= 0.2) & ~face
    shadow = (shift(face, 1, 1) | shift(face, 0, 1)) & ~face
    idx = np.zeros(a.shape, np.uint8)
    idx[shadow] = 2
    cols = np.where(face.any(0))[0]; c0, c1 = cols[0], cols[-1]
    ramp = [11, 9, 10, 12, 13, 14]
    xx = (np.arange(a.shape[1]) - c0) / max(1, c1 - c0)
    g = np.array([ramp[min(len(ramp) - 1, int(v * len(ramp)))] if 0 <= v <= 1 else 11 for v in xx], np.uint8)
    rows = np.where(face.any(1))[0]
    gi = np.repeat(g[None, :], a.shape[0], 0)
    idx[face] = gi[face]
    idx[edge & ~shadow & ~face & shift(face, -1, 0)] = 7
    ys, xs = np.where(idx > 0)
    return idx[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def build_art(tl, problems, d=None, px=None):
    if d is None: d, px = em_load()
    out = {}
    # header
    hp = header_parts(tl.get('hdr', 'EDIT Character Register'))
    h = header_art(hp)
    if h.shape[0] > HDR_H or h.shape[1] > 510:
        problems.append(f'header {h.shape[1]}x{h.shape[0]} > 510x{HDR_H}')
    else:
        # like the JP sprites: columns 0-254 at texel 512, the rest from
        # texel 768 (texel 767 is skipped; sprite widths stay <= 255)
        px[HDR_Y:HDR_Y + HDR_H, 512:1024] = 0
        y0 = HDR_Y + (HDR_H - h.shape[0]) // 2
        w = h.shape[1]
        px[y0:y0 + h.shape[0], 512:512 + min(w, 255)] = h[:, :255]
        if w > 255: px[y0:y0 + h.shape[0], 768:768 + w - 255] = h[:, 255:]
        out['hdr'] = h
    # school labels
    for i, (sx, sy) in enumerate(SCH_SLOTS):
        s = school_art(tl.get(f'sch{i}', ''))
        if s.shape[1] > SCH_W or s.shape[0] > SCH_H:
            problems.append(f'sch{i} {s.shape[1]}x{s.shape[0]} > {SCH_W}x{SCH_H}'); continue
        px[sy:sy + SCH_H, sx:sx + SCH_W] = 0
        y0 = sy + (SCH_H - s.shape[0]) // 2
        px[y0:y0 + s.shape[0], sx:sx + s.shape[1]] = s
        out[f'sch{i}'] = s
    return d, px, out

def header_parts(text):
    """'EDIT ...' gets a larger EDIT, the rest one size."""
    if text.startswith('EDIT '): return [('EDIT', 29), (' ' + text[5:], 24)]
    return [(text, 26)]

def patch_kanri_art(k, art):
    if 'hdr' in art:
        w = art['hdr'].shape[1]; x0 = HDR_CX - w // 2
        wa = min(w, 255); wb = max(1, w - 255)
        for off, tpx, ww, x in ((HDR_REC[0], 768, wa, x0), (HDR_REC[1], 832, wb, x0 + 255)):
            r = list(struct.unpack_from('<11h', k, off))
            r[1], r[5], r[6], r[7], r[8], r[9], r[10] = tpx, 0, HDR_Y, ww, HDR_H, x, 32
            struct.pack_into('<11h', k, off, *r)
    for i, (sx, sy) in enumerate(SCH_SLOTS):
        if f'sch{i}' not in art: continue
        tpx = 640 + (sx // 256) * 64
        struct.pack_into('<8h', k, SCH_REC + 16 * i, 0, tpx, 256, sx % 256, sy, SCH_W, SCH_H, 0)
    return k

def full_patch(k, problems=None):
    """Strings and art records on a KANRI image (for build_editcm.py)."""
    problems = [] if problems is None else problems
    tl = load_tl()
    k = patch_kanri(k, tl, problems)
    _, _, art = build_art(tl, problems)
    return patch_kanri_art(k, art)

# ---------------------------------------------------------------- main
def main(argv):
    check = '--check' in argv; png = '--png' in argv
    tl = load_tl(); problems = []
    src = KANRI_OUT if os.path.exists(KANRI_OUT) else KANRI_DISC
    k = patch_kanri(bytearray(open(src, 'rb').read()), tl, problems)
    d, px, art = build_art(tl, problems)
    k = patch_kanri_art(k, art)
    em_store(d, px)
    slot = os.path.getsize(EM_DISC); slot += -slot % 2048
    for cap in (0xe00, 0xe40, 0xe80):
        c = lz.compress(bytes(d), 256, cap)
        per = burst.bursts(c); last = max(per)
        mb = max((v for s, v in per.items() if s < last - 1), default=0)
        if len(c) <= slot and mb <= BURST: break
    if len(c) > slot: problems.append(f'EDITMOJI compressed {len(c)} > slot {slot}')
    if mb > BURST: problems.append(f'EDITMOJI burst {mb:#x} > {BURST:#x}')
    if lz.decompress(c)[0][:len(d)] != bytes(d): problems.append('EDITMOJI LZ round trip failed')
    print(f'KANRI: {len(k):#x} bytes (from {src}); EDITMOJI: {len(c)} bytes (slot {slot}), burst {mb:#x}')
    for p in problems: print('  ' + p)
    if png:
        os.makedirs('private/re/register/gfx', exist_ok=True)
        for name in ['hdr'] + [f'sch{i}' for i in range(4)]:
            a = art.get(name); cx = 112 if name == 'hdr' else 288
            if a is None: continue
            P = np.array([((c & 31) << 3, (c >> 5 & 31) << 3, (c >> 10 & 31) << 3, 255 if j else 0)
                          for j, c in enumerate(em_clut(d, cx))], np.uint8)
            Image.fromarray(P[a]).save(f'private/re/register/gfx/{name}.png')
    if not check and not problems:
        open(KANRI_OUT, 'wb').write(k)
        open(EM_OUT, 'wb').write(c)
        open(EM_OUT + '.lz', 'wb').write(bytes(d))
    print(f'KANRI: problems {len(problems)}')
    return 1 if problems else 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
