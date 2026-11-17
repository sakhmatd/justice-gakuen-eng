# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Exam image text in English: CEFT4.BIZ (TEST/START, TIME/UP, HUD "Subj"),
SIKEN2.BIZ (results sheet, comments, subject names, practice menu, stamps)
and SIKEN1.BIZ (Exam Practice menus).

Reads tl/EXAMGFX.txt (ID: EN, '|' = line break) and the lesson titles
(title: lines) in tl/EXAM.txt. Inputs: private/unpacked/*.BIZ.lz for the
images, private/build/SIKEN.BIN (after build_exam.py), private/build/SIKENGO1/2.BIN
if present, else the disc copies. Writes private/build/CEFT4.BIZ, SIKEN1.BIZ,
SIKEN2.BIZ (+ .lz), SIKEN.BIN, SIKENGO1.BIN, SIKENGO2.BIN. nick.py runs
after this and patches other SIKEN.BIN bytes (0x1db0-0x1dfb).
--png writes before/after images and screen mock-ups to private/re/exam/en/.

The atlases are cleared except for the pieces that stay (digits, grades,
brackets, cursor); every English string is packed in as one strip and its
sprite list is rewritten in place with at most the original piece count. Strips longer than 255
texels (or than any free run) are cut into adjacent pieces; the game
places the pieces of a list edge to edge. RE: private/re/exam/notes.txt.
"""
import sys, os, struct, collections
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import lz, streamsim, burst
from build_exam import parse as parse_exam

FONT = '/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf'
FONT_BIG = '/usr/share/fonts/TTF/impact.ttf'
FONT_STAMP = '/usr/share/fonts/TTF/ariblk.ttf'
B = 0x8009c000                     # SIKEN.BIN load address
GO_BASE = {'SIKENGO1.BIN': 0x8014c000, 'SIKENGO2.BIN': 0x80174000}
PNG = 'private/re/exam/en'
HUD_DIGIT_X, HUD_DIGIT_END = 70, 91   # exam HUD digit cell x, last ink column
BURST = {'SIKEN1.BIZ': 0x1600}     # fallback cap if the default 0xe00 overflows the slot

# ---------------------------------------------------------------- TIM ---

class Tim:
    """4bpp or 8bpp TIM inside a decompressed buffer."""
    def __init__(self, buf, off):
        self.buf, self.off = buf, off
        flag, bl = struct.unpack_from('<II', buf, off + 4)
        self.bpp = 4 if flag & 3 == 0 else 8
        self.cx, self.cy, cw, ch = struct.unpack_from('<4H', buf, off + 12)
        self.clut = struct.unpack_from(f'<{cw * ch}H', buf, off + 20)
        p = off + 8 + bl
        _, self.vx, self.vy, w, h = struct.unpack_from('<IHHHH', buf, p)
        self.pix, self.stride, self.H = p + 12, w * 2, h
        raw = np.frombuffer(bytes(buf[self.pix:self.pix + w * 2 * h]), np.uint8).reshape(h, w * 2)
        if self.bpp == 4:
            self.px = np.empty((h, w * 4), np.uint8)
            self.px[:, 0::2] = raw & 15; self.px[:, 1::2] = raw >> 4
        else:
            self.px = raw.copy()
        self.W = self.px.shape[1]
        self.orig = self.px.copy()

    def store(self):
        if self.bpp == 4:
            raw = (self.px[:, 0::2] | (self.px[:, 1::2] << 4)).astype(np.uint8)
        else:
            raw = self.px
        self.buf[self.pix:self.pix + self.stride * self.H] = raw.tobytes()

    def pal(self, k):
        n = 16 if self.bpp == 4 else 256
        return self.clut[k * n:(k + 1) * n]

    def rgb(self, k):
        """(n, 4) RGBA; CLUT value 0 is transparent."""
        return np.array([((v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3, 255 if v else 0)
                         for v in self.pal(k)], np.uint8)

# ---------------------------------------------------------- rendering ---

_fonts = {}
def font(path, size):
    if (path, size) not in _fonts: _fonts[path, size] = ImageFont.truetype(path, size)
    return _fonts[path, size]

def line_alpha(text, h, size, path=FONT, maxw=None, minsx=0.72, ss=4, base=None, sx=None):
    """Alpha (h, w) of one line. The baseline sits so that capitals are
    centred in h (or at row `base`). Squeezes horizontally down to minsx,
    then shrinks the size, to fit maxw. Width = the advance width."""
    fixed = sx
    while True:
        f = font(path, size * ss)
        adv = f.getlength(text) / ss
        if fixed: break
        sx = 1.0 if not maxw or adv <= maxw else maxw / adv
        if sx >= minsx or size <= 8: break
        size -= 1
    hb = f.getbbox('H', anchor='ls')
    cap = (hb[3] - hb[1]) / ss
    by = base if base is not None else (h + cap) / 2
    W = int(np.ceil(adv)) + 1
    im = Image.new('L', (W * ss, h * ss), 0)
    ImageDraw.Draw(im).text((0, round(by * ss)), text, font=f, fill=255, anchor='ls')
    w = max(1, int(np.ceil(adv * sx)))
    im = im.resize((max(w, int(round(W * sx))), h), Image.BOX)
    return (np.asarray(im, np.float32) / 255)[:, :w]

def group_sx(texts, size, maxw, path=FONT):
    """One horizontal scale that fits the widest of texts into maxw."""
    w = max(font(path, size * 4).getlength(t) / 4 for t in texts)
    return min(1.0, maxw / w)

def nearest(rgb, pal, cand):
    cand = list(cand)
    P = np.array([pal[i][:3] for i in cand], float)
    d = ((rgb[..., None, :] - P) ** 2).sum(-1)
    return np.array(cand, np.uint8)[d.argmin(-1)]

def blend_idx(a, fg, bg, pal, cand, cut=0.1):
    """Index of the CLUT colour nearest to bg..fg blended by alpha; 0 below cut."""
    fg, bg = np.array(fg, float), np.array(bg, float)
    rgb = bg + (fg - bg) * a[..., None]
    out = nearest(rgb, pal, cand)
    out[a < cut] = 0
    return out

def pad(a, l=1, r=1, t=0, b=0):
    return np.pad(a, ((t, b), (l, r)))

def shift(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out

# Styles: each turns alpha into palette indices for one palette.
def st_sheet(pal, red=False):
    """Results sheet text: dark grey (or the subject red) antialiased to the
    white sheet; index 0 transparent."""
    if red: return lambda a: blend_idx(a, pal[3][:3], (248, 248, 248), pal, (15, 14, 12, 10, 8, 6, 4, 3))
    return lambda a: blend_idx(a, pal[1][:3], (248, 248, 248), pal, (15, 13, 11, 9, 7, 5, 2, 1))

def st_menu(pal):
    """Practice menu white text with a soft dark shadow to the lower right."""
    greys = list(range(2, 15))
    def f(a):
        a = pad(a, 0, 1, 0, 0)
        sh = np.maximum(shift(a, 1, 1), shift(a, 0, 1) * 0.6)
        out = np.zeros(a.shape, np.uint8)
        out[sh > 0.3] = 4
        t = blend_idx(a, (248, 248, 248), (40, 40, 40), pal, greys, cut=0.25)
        return np.where(t > 0, t, out)
    return f

def st_yellow(pal):
    """Menu header: yellow fill, darker yellow edge, black outline."""
    def f(a):
        a = pad(a, 2, 2, 0, 0)
        o = ndimage.grey_dilation(a, footprint=np.ones((3, 3)))
        o2 = np.maximum(o, shift(o, 1, 1))
        out = np.zeros(a.shape, np.uint8)
        out[o2 > 0.35] = 15
        out[o > 0.35] = 13
        out[a > 0.3] = 10
        out[a > 0.5] = 9
        out[a > 0.65] = 8
        return out
    return f

def st_hud(a):
    """HUD label in CEFT4 pal 8, a ramp from 1 (white) to 15 (dark blue):
    1 px white outline, fill 15, anti-aliased edge through the ramp, as in
    the original digits."""
    a = np.pad(a, 1)
    o = ndimage.grey_dilation(a, footprint=np.ones((3, 3))) > 0.3
    out = np.zeros(a.shape, np.uint8)
    out[o] = 1
    m = a > 0.12
    out[m] = np.clip(np.rint(1 + a[m] * 14), 2, 15)
    return out[1:-1]

def hud_alpha(text, size, base, h=16, shear=0.0, ss=4, path=FONT):
    """Alpha (h, w) with the baseline at row `base`, slanted by `shear`
    (x per row), cropped to ink."""
    f = font(path, size * ss)
    W = (int(f.getlength(text)) // ss + 8 + int(shear * h)) * ss
    im = Image.new('L', (W, h * ss), 0)
    ImageDraw.Draw(im).text((4 * ss, base * ss), text, font=f, fill=255, anchor='ls')
    if shear:
        im = im.transform(im.size, Image.AFFINE, (1, shear, -shear * base * ss, 0, 1, 0), Image.BICUBIC)
    a = np.asarray(im.resize((W // ss, h), Image.BOX), np.float32) / 255
    cols = np.nonzero((a > 0.05).any(0))[0]
    return a[:, cols.min():cols.max() + 1]

# ------------------------------------------------------------- texels ---

class Atlas:
    """Free-texel bookkeeping. Strips never cross a 256-texel tpage."""
    def __init__(self, tim, free, tpw=256):
        self.t, self.free, self.tpw = tim, free.copy(), tpw

    def _fits(self, w, h):
        F = self.free.astype(np.int32)
        S = np.zeros((F.shape[0] + 1, F.shape[1] + 1), np.int32)
        S[1:, 1:] = F.cumsum(0).cumsum(1)
        win = S[h:, w:] - S[:-h, w:] - S[h:, :-w] + S[:-h, :-w]
        ok = win == w * h
        xs = np.arange(ok.shape[1])
        ok &= ((xs % self.tpw) + w <= self.tpw)[None, :]
        return ok

    def alloc(self, w, h):
        ok = self._fits(w, h)
        ys, xs = np.nonzero(ok)
        if not len(ys): return None
        y, x = int(ys[0]), int(xs[0])
        self.free[y:y + h, x:x + w] = False
        return x, y

    def put(self, strip, maxpieces=1):
        """Place strip (h, w) as up to maxpieces adjacent pieces; returns
        [(x, y, w)] in drawing order or None."""
        h, W = strip.shape
        if W <= self.tpw:
            p = self.alloc(W, h)
            if p:
                self._draw(strip, p[0], p[1]); return [(p[0], p[1], W)]
        if maxpieces <= 1: return None
        # cut: take the widest run that fits, then place the rest
        for w in range(min(W, self.tpw) - 1, 7, -1):
            ok = self._fits(w, h)
            ys, xs = np.nonzero(ok)
            if not len(ys): continue
            y, x = int(ys[0]), int(xs[0])
            self.free[y:y + h, x:x + w] = False
            rest = self.put(strip[:, w:], maxpieces - 1)
            if rest is None:
                self.free[y:y + h, x:x + w] = True; return None
            self._draw(strip[:, :w], x, y)
            return [(x, y, w)] + rest
        return None

    def pack(self, items, problems, what):
        """items: (key, strip, max pieces). Taller and wider strips first."""
        got, width = {}, {}
        for key, strip, budget in sorted(items, key=lambda i: (-i[1].shape[0], -i[1].shape[1])):
            r = self.put(strip, budget)
            if r is None:
                problems.append(f'{what}: no room for {key} ({strip.shape[1]}x{strip.shape[0]})'); r = []
            got[key], width[key] = r, strip.shape[1]
        return got, width

    def _draw(self, strip, x, y):
        h, w = strip.shape
        self.t.px[y:y + h, x:x + w] = strip

# ------------------------------------------------------------ text io ---

def load_tl(path='tl/EXAMGFX.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'): continue
        k, v = line.split(':', 1)
        out[k.strip()] = v[1:] if v.startswith(' ') else v
    lessons, _ = parse_exam()
    for L in range(30):
        out[f'subj{L + 1:02d}'] = lessons[L]['title']
    return out

def u32(d, o): return struct.unpack_from('<I', d, o)[0]

# --------------------------------------------------------- list codec ---

def s2_lists(d, ptab, n, base):
    """SIKEN2-style lists: u8 n, n x (tpx/64, tpy/256, clx/16, cly-480, u, v, w, h)."""
    out = []
    for i in range(n):
        p = u32(d, ptab + 4 * i) - base
        k = d[p]
        out.append((p, [tuple(d[p + 1 + 8 * j:p + 9 + 8 * j]) for j in range(k)]))
    return out

def s1_lists(d, ptab=0xf50, n=46):
    """SIKEN1 menu lists: u8 n, n x (tp/64, u, v, w, h)."""
    out = []
    for i in range(n):
        p = u32(d, ptab + 4 * i) - B
        k = d[p]
        out.append((p, [tuple(d[p + 1 + 5 * j:p + 6 + 5 * j]) for j in range(k)]))
    return out

def write_list(d, p, pieces, size, budget):
    assert len(pieces) <= budget, (hex(p), len(pieces), budget)
    d[p] = len(pieces)
    for j, e in enumerate(pieces):
        d[p + 1 + size * j:p + 1 + size * (j + 1)] = bytes(e)

def s2_rect(e):
    """Atlas texel rect of a SIKEN2 atlas piece (TIM at VRAM 768,0)."""
    tpx, tpy, cx, cy, u, v, w, h = e
    return (tpx * 64 - 768) * 4 + u, tpy * 256 + v, w, h

def s2_piece(x, y, w, h):
    return (12 + x // 256, 0, 0, 2, x % 256, y, w, h)

def pm_piece(x, y, w, h):         # practice menu TIM at VRAM (896,384)
    return (14, 1, 1, 3, x, 128 + y, w, h)

def s1_piece(x, y, w, h):
    return (12 + x // 256, x % 256, y, w, h)

def set_imm(d, ram, val, base=B):
    o = ram - base
    w = u32(d, o)
    assert w >> 16 in (0x2406, 0x2407), (hex(ram), hex(w))
    struct.pack_into('<I', d, o, (w & 0xffff0000) | (val & 0xffff))

# ------------------------------------------------------------- CEFT4 ---

def big_start(t, rect, lines):
    """開始/終了 style: red->yellow fill by row (from the original), thin dark
    outline, cyan drop shadow to the lower right."""
    x0, y0, W, H = rect
    orig = t.orig[y0:y0 + H, x0:x0 + W]
    fills = (4, 6, 7, 8, 9, 10)
    rowfill = []
    for r in range(H):
        c = collections.Counter(int(v) for v in orig[r] if v in fills)
        rowfill.append(c.most_common(2) if c else None)
    # carry the nearest known row into rows without fill
    known = [r for r in range(H) if rowfill[r]]
    a = np.zeros((H, W), np.float32)
    n = len(lines)
    pitch = H // n
    sx = group_sx(lines, 40, W - 6, FONT_BIG)
    for i, s in enumerate(lines):
        la = line_alpha(s, pitch, 40, FONT_BIG, base=pitch - 5, sx=sx)
        lx = (W - 4 - la.shape[1]) // 2
        a[i * pitch:(i + 1) * pitch, lx:lx + la.shape[1]] = np.maximum(
            a[i * pitch:(i + 1) * pitch, lx:lx + la.shape[1]], la)
    m = a > 0.45
    o = ndimage.binary_dilation(m, np.ones((3, 3), bool))
    sh = shift(o, 2, 3) | shift(o, 1, 2) | shift(o, 2, 2) | shift(o, 1, 1)
    out = np.zeros((H, W), np.uint8)
    out[sh] = 15
    out[sh & ~shift(sh, 1, 1)] = 14
    for r in range(H):
        out[r, o[r]] = 1 if r < H // 2 else 3
    for r in range(H):
        k = rowfill[r] or rowfill[min(known, key=lambda q: abs(q - r))]
        a_, b_ = k[0][0], (k[1][0] if len(k) > 1 else k[0][0])
        pb = (k[1][1] / (k[0][1] + k[1][1])) if len(k) > 1 else 0
        for c in np.nonzero(m[r])[0]:
            out[r, c] = b_ if (pb > 0.3 and (r + c) % 2 == 0) else a_
    t.px[y0:y0 + H, x0:x0 + W] = out

def build_ceft4(tl, siken, prev):
    d = bytearray(open('private/unpacked/CEFT4.BIZ.lz', 'rb').read())
    t = Tim(d, 0x10800)
    big_start(t, (256, 0, 80, 96), tl['start'].split('|'))
    big_start(t, (336, 0, 80, 96), tl['end'].split('|'))
    # HUD (SIKEN 0x800a2630, y 190): label, digit at x 70 (24x16 cell,
    # pal 8, ink rows 0-11, digits 2-5 end at x 89-91), ':' (pal 7, ink
    # cols 9-15 of its cell), title at the x immediate at 0x800a2a2c.
    # The label is as tall as the digit (caps on rows 1-10), slanted like
    # it, and ends 1 px before the digit cell. It sits at texel (184,112)
    # over the old kanji cells; the ':' moves into the old 科 cell.
    colon = t.orig[112:128, 208:232].copy()
    t.px[96:128, 184:232] = 0
    t.px[96:112, 184:208] = colon
    s = st_hud(hud_alpha(tl['hud_subject'], 14, 11, shear=0.2))
    w = s.shape[1]
    assert w <= 48 and not s[15].any(), w
    t.px[112:128, 184:184 + w] = s
    lx = max(30, HUD_DIGIT_X + 1 - w)
    cx = HUD_DIGIT_END + 2 - 9                 # 2 px after the digit ink
    tx = cx + 16 + 3                           # 3 px after the ':' ink
    # records (tp/64, clx/16, cly-500, u, v, w, h, x-30): one draws the
    # label, the second is a copy of it (same place), the third is the ':'
    siken[0x17c0:0x17c8] = bytes([12, 8, 0, 184, 112, w, 16, lx - 30])
    siken[0x17c8:0x17d0] = bytes([12, 8, 0, 184, 112, w, 16, lx - 30])
    siken[0x17d0:0x17d8] = bytes([12, 7, 0, 184, 96, 24, 16, cx - 30])
    # the TIME timer's two ':' records shared the old cell (208,112), which
    # now holds the "j" of "Subj" (private/re/exam/findings-time.org)
    siken[0x179f:0x17a1] = bytes([184, 96])
    siken[0x17a7:0x17a9] = bytes([184, 96])
    o = 0x800a2a2c - B                         # addiu v0, zero, title x
    assert u32(siken, o) >> 16 == 0x2402, hex(u32(siken, o))
    struct.pack_into('<H', siken, o, tx)
    t.store()
    prev['CEFT4'] = (t, [(256, 0, 160, 96, 2), (184, 96, 48, 32, 8)])
    return d

# ------------------------------------------------------------ SIKEN2 ---

def build_siken2(tl, siken, gos, prev, problems):
    d = bytearray(open('private/unpacked/SIKEN2.BIZ.lz', 'rb').read())
    t = Tim(d, 0x4800)                       # results atlas 768x256
    pm = Tim(d, 0x2000)                      # practice menu 256x64
    pal, pmpal = t.rgb(0), pm.rgb(1)
    disc = open('private/disc/DAT1/SIKEN.BIN', 'rb').read()
    dgo = open('private/disc/DAT1/SIKENGO1.BIN', 'rb').read()
    R = s2_lists(disc, 0x1310, 60, B)
    G = s2_lists(dgo, 0xa7c, 60, GO_BASE['SIKENGO1.BIN'])
    C = s2_lists(dgo, 0xcfc, 11, GO_BASE['SIKENGO1.BIN'])
    hdr = [list(struct.unpack_from('<9h', dgo, 0x12 + 18 * i)) for i in range(10)]

    # --- results atlas: clear everything except the digits, the grade
    # letters A-E and S. The remaining Japanese pieces (comment fragments,
    # a second の/が, 合格だ) are referenced by no list in SIKEN, SIKENGO1/2
    # or any other DAT1 file and are dropped.
    keep = [s2_rect(e) for i in list(range(30, 40)) + list(range(41, 46)) for e in R[i][1]]
    keep += [s2_rect(e) for e in G[56][1]]
    free = np.ones(t.px.shape, bool)
    t.px[:] = 0
    for x, y, w, h in keep:
        t.px[y:y + h, x:x + w] = t.orig[y:y + h, x:x + w]; free[y:y + h, x:x + w] = False
    A = Atlas(t, free)
    dark, red = st_sheet(pal), st_sheet(pal, red=True)

    def strip(text, h, size, style, maxw=None, sx=None):
        return style(pad(line_alpha(text, h, size, FONT, maxw=maxw, minsx=0.55, sx=sx), 0, 1))

    items = [('res_header', strip(tl['res_header'], 32, 24, dark, maxw=150), 1)]
    LABELS = ('res_col_subject', 'res_col_action', 'res_col_accuracy', 'res_col_speed', 'res_col_grade')
    lsx = group_sx([tl[k] for k in LABELS[1:]], 16, 56)
    for k in LABELS:
        items.append((k, strip(tl[k], 24, 16, dark, sx=lsx), 1))
    for L in range(30):
        items.append((L, strip(tl[f'subj{L + 1:02d}'], 24, 16, red, maxw=150), 1))
    items.append((40, strip(tl['res_total'], 24, 16, dark), 1))
    titles = [tl['res_t_nyugaku'], tl['menu_exam3'], tl['menu_exam4'], tl['menu_exam5'],
              tl['menu_exam6'], tl['res_t_gakunen']]
    for k, ttl in enumerate(titles):
        items.append((49 + k, strip(ttl, 24, 16, dark, maxw=150), 1))
    items.append(('c0', strip(tl['res_no'], 24, 16, dark), 1))
    items.append(('c4', strip(tl['res_ga'], 24, 16, dark), 1))
    for k in range(6):
        items.append((f'c{5 + k}', strip(tl[f'comment{k}'], 19, 15, dark, maxw=356), len(C[5 + k][1])))
    got, width = A.pack(items, problems, 'SIKEN2')
    lists = {}
    for i in list(range(30)) + [40] + list(range(49, 55)):
        lists[i] = [s2_piece(x, y, w, 24) for x, y, w in got[i]]
    tw = [width[49 + k] for k in range(6)]
    # header table (s16 tpx, tpy, cly, u, v, w, h, x, y), same in GO1/GO2
    newhdr = [h[:] for h in hdr]
    def hrec(i, p, h, x, y):
        if not p: return
        (px_, py_, w), = p
        newhdr[i][:7] = [768 + 64 * (px_ // 256), 0, 482, px_ % 256, py_, w, h]
        newhdr[i][7:9] = [x, y]
    cols = {'res_col_action': (231, 290), 'res_col_accuracy': (290, 349),
            'res_col_speed': (349, 405), 'res_col_grade': (405, 462)}
    hrec(0, got['res_header'], 32, max(224, 80 + max(tw) + 12), 21)
    for i, k in enumerate(LABELS):
        if k == 'res_col_subject': hrec(i + 1, got[k], 24, 64, 71)
        else:
            a_, b_ = cols[k]; hrec(i + 1, got[k], 24, (a_ + b_ - width[k]) // 2 + 1, 70)
    # comment lists C00-C10 (stat labels reuse the header texels)
    clists = {0: [s2_piece(x, y, w, 24) for x, y, w in got['c0']],
              4: [s2_piece(x, y, w, 24) for x, y, w in got['c4']]}
    for c, k in ((1, 'res_col_action'), (2, 'res_col_accuracy'), (3, 'res_col_speed')):
        clists[c] = [s2_piece(x, y, w, 24) for x, y, w in got[k]]
    for k in range(6):
        clists[5 + k] = [s2_piece(x, y, w, 19) for x, y, w in got[f'c{5 + k}']]

    # --- practice menu TIM (256x64): keep the unreferenced row-1/2 pieces
    pr_free = np.zeros(pm.px.shape, bool)
    for x, y, w, h in ((0, 0, 256, 20), (0, 20, 160, 20), (180, 40, 76, 20)):
        pm.px[y:y + h, x:x + w] = 0; pr_free[y:y + h, x:x + w] = True
    PA = Atlas(pm, pr_free)
    pr = {}
    for li, k in ((46, 'pr_again'), (47, 'pr_other'), (48, 'pr_title'), (55, 'pr_back')):
        a = line_alpha(tl[k], 20, 15, FONT)
        s = st_menu(pmpal)(a)
        r = PA.put(s, len(R[li][1]))
        if r is None: problems.append(f'SIKEN2: no room for {k}'); r = []
        lists[li] = [pm_piece(x, y, w, 20) for x, y, w in r]
        pr[li] = s.shape[1] + 0
    # practice menu positions (SIKEN file 0x2c: 4 x (u16 x, y)); keep the
    # centre of each item. Entries 0, 1, 3 draw lists 46, 47, 55 (SIKEN
    # 0x800a1250); entry 2 is taken to belong to list 48 (186 + 140/2 = 256).
    ow = {li: sum(e[6] for e in R[li][1]) for li in (46, 47, 48, 55)}
    for k, li in ((0, 46), (1, 47), (2, 48), (3, 55)):
        ox = struct.unpack_from('<H', disc, 0x2c + 4 * k)[0]
        struct.pack_into('<H', siken, 0x2c + 4 * k, ox + (ow[li] - pr[li]) // 2)

    # --- write the lists: SIKEN (R) and SIKENGO1/2 (G, C, header, title x)
    done = set()
    for i, pieces in lists.items():
        p, old = R[i]
        if p in done: continue
        done.add(p); write_list(siken, p, pieces, 8, len(old))
    for name, g in gos.items():
        dg = open(f'private/disc/DAT1/{name}', 'rb').read()
        GG = s2_lists(dg, 0xa7c, 60, GO_BASE[name])
        CC = s2_lists(dg, 0xcfc, 11, GO_BASE[name])
        seen = set()
        for i, pieces in lists.items():
            p, old = GG[i]
            if p in seen: continue
            seen.add(p); write_list(g, p, pieces, 8, len(old))
        for c, pieces in clists.items():
            p, old = CC[c]; write_list(g, p, pieces, 8, len(old))
        for i, h in enumerate(newhdr):
            struct.pack_into('<9h', g, 0x12 + 18 * i, *h)
    t.store(); pm.store()

    # --- logo tile behind the sheet
    lg = Tim(d, 0)
    logo_tile(lg.px); lg.store(); prev['logo'] = (lg, None)

    # --- stamps
    for off, key in ((0x31000, 'stamp_pass'), (0x44800, 'stamp_fail')):
        st = Tim(d, off)
        stamp(st, tl[key])
        st.store()
        prev[key] = (st, None)
    prev['SIKEN2'] = (t, None); prev['SIKEN2pm'] = (pm, None)
    return d, lists, clists, newhdr

# -------------------------------------------------------------- logo ---

# Background logo tile (SIKEN2 TIM 0x0, 4bpp 144x96, 7 CLUTs at (0,510),
# one per colour scheme; the same index ramp in each: 1-5 dark, 6 the
# background, 8-f light). The identical TIM also sits in MENU.BIZ 0x50000,
# MZBG13.BIZ 0x26800, SELP3.BIZ 0x6800, MZBG03/07/12.BIZ entry 0 0x1b000
# and MZPANEL.BIZ entries 0-2 0x41000 (offsets in the unpacked streams);
# logo_tile() can be reused for those.
LOGO = ('RIVAL SCHOOLS', 'JUSTICE', 'HIGH')
LOGO_KEEP_Y = 64          # rows from here down (LEGION OF HEROES plate) stay

def logo_tile(px):
    """Redraw the 144x96 index array px in place: the 私立ジャスティス学園
    lettering becomes LOGO in a slanted heavy face with a light gradient
    fill, dark outline and drop shadow; the LEGION OF HEROES plate stays."""
    H, W = px.shape
    ss = 4
    keep = px.copy()
    px[:LOGO_KEEP_Y] = 6
    px[LOGO_KEEP_Y:, :28] = 6; px[LOGO_KEEP_Y:, 116:] = 6
    lay = Image.new('L', (W * ss, H * ss), 0)
    dr = ImageDraw.Draw(lay)
    def word(text, x, base, cap, sx=1.0, maxw=None):
        size = int(cap / 0.79 * ss)       # Impact cap height ~0.79 em
        f = font(FONT_BIG, size)
        w = f.getlength(text)
        if maxw and w * sx > maxw * ss: sx = maxw * ss / w
        im = Image.new('L', (int(w) + 8, H * ss), 0)
        ImageDraw.Draw(im).text((0, base * ss), text, font=f, fill=255, anchor='ls')
        im = im.resize((max(1, int(im.width * sx)), im.height), Image.LANCZOS)
        lay.paste(Image.fromarray(np.maximum(np.asarray(lay.crop((x * ss, 0, x * ss + im.width, H * ss))),
                                             np.asarray(im))), (x * ss, 0))
        return x + im.width / ss
    word(LOGO[0], 24, 14, 10, sx=0.95, maxw=96)
    end = word(LOGO[1], 8, 55, 35, sx=0.80, maxw=96)
    word(LOGO[2], int(end) + 2, 47, 22, sx=0.80, maxw=W - 8 - int(end))
    k, yc = 0.22, 40 * ss
    lay = lay.transform(lay.size, Image.AFFINE, (1, k, -k * yc, 0, 1, 0), Image.BICUBIC)
    a = np.asarray(lay.resize((W, H), Image.BOX), np.float32) / 255
    m = a > 0.45
    # backing banner: a slanted band from the title down to the plate,
    # halftone mid tone with a dark rim (stands in for the original wings)
    ban = Image.new('L', (W * ss, H * ss), 0)
    ImageDraw.Draw(ban).polygon([(14 * ss, 22 * ss), (136 * ss, 17 * ss), (124 * ss, 60 * ss),
                                 (100 * ss, 66 * ss), (40 * ss, 66 * ss), (2 * ss, 60 * ss)], fill=255)
    b = np.asarray(ban.resize((W, H), Image.BOX), np.float32) / 255 > 0.5
    yy, xx = np.mgrid[0:H, 0:W]
    px[b] = np.where(((yy + xx) % 2 == 0)[b], 8, 7)
    px[b & ~ndimage.binary_erosion(b, np.ones((3, 3), bool))] = 2
    o1 = ndimage.binary_dilation(m, np.ones((3, 3), bool))
    o2 = ndimage.binary_dilation(o1, ndimage.generate_binary_structure(2, 1))
    sh = shift(o2, 2, 2) & ~o2
    px[sh & (px == 6)] = 4
    px[o2] = 2
    px[o1] = 1
    # fill: light at the top of each word band, darker below, 2x2 dither
    ramp = (15, 14, 13, 12, 11, 10, 9)
    yy, xx = np.nonzero(m)
    for y, x in zip(yy, xx):
        band = (2, 16) if y < 17 else (17, 60)
        t_ = (y - band[0]) / (band[1] - band[0])
        v = t_ * (len(ramp) - 1)
        i = int(v); fr = v - i
        thr = (0.25, 0.75, 1.0, 0.5)[(y & 1) * 2 + (x & 1)]
        px[y, x] = ramp[min(len(ramp) - 1, i + (1 if fr > thr else 0))]
    # light edge on the upper-left of strokes (like the original's sheen)
    hl = m & ~shift(m, 1, 1)
    px[hl] = 15
    # keep the plate where the text does not cover it
    plate = np.zeros_like(m); plate[LOGO_KEEP_Y:, 28:116] = True
    plate &= keep != 6
    px[plate & ~o2] = keep[plate & ~o2]

# ------------------------------------------------------------- stamp ---

def stamp(t, text):
    """Rubber stamp: rounded double frame and the word along the long axis
    of the original's rotated rectangle. Ink 14, antialias 13..1 (to
    white), background 15 (transparent). The last 4 columns held stray
    texels (visible in game) and are cleared."""
    W = t.W - 4
    ink = t.orig[:, :W] != 15
    ys, xs = np.nonzero(ink)
    c = np.array([xs.mean(), ys.mean()])
    cov = np.cov(np.vstack([xs - c[0], ys - c[1]]))
    ev, evec = np.linalg.eigh(cov)
    ax = evec[:, 1]
    if ax[1] > 0: ax = -ax                    # long axis pointing up
    ang = np.degrees(np.arctan2(-ax[1], ax[0]))   # text baseline angle, ccw
    pr = np.stack([xs - c[0], ys - c[1]], 1)
    u = pr @ ax; v = pr @ np.array([-ax[1], ax[0]])
    L = np.percentile(u, 99.7) - np.percentile(u, 0.3)
    S = np.percentile(v, 99.7) - np.percentile(v, 0.3)
    ss = 3
    Lp, Sp = int(L * ss), int(S * ss)
    im = Image.new('L', (Lp, Sp), 0)
    dr = ImageDraw.Draw(im)
    th = int(S * 0.09 * ss); r = int(S * 0.16 * ss)
    dr.rounded_rectangle((th // 2, th // 2, Lp - th // 2, Sp - th // 2), radius=r, outline=255, width=th)
    g = int(S * 0.15 * ss); th2 = max(ss, int(th * 0.45))
    dr.rounded_rectangle((g, g, Lp - g, Sp - g), radius=int(r * 0.6), outline=255, width=th2)
    inner_w, inner_h = Lp - 2 * g - 6 * th2, Sp - 2 * g - 6 * th2
    size = inner_h
    while True:
        f = font(FONT_STAMP, size)
        b = f.getbbox(text)
        if (b[2] - b[0] <= inner_w and b[3] - b[1] <= inner_h * 0.9) or size < 10: break
        size -= 2
    tw_, th_ = b[2] - b[0], b[3] - b[1]
    dr.text(((Lp - tw_) // 2 - b[0], (Sp - th_) // 2 - b[1]), text, font=f, fill=255)
    # wear: a few soft gaps in the ink, deterministic
    rng = np.random.default_rng(7)
    a = np.asarray(im, np.float32) / 255
    noise = ndimage.gaussian_filter(rng.random(a.shape), 6 * ss / 3)
    noise = (noise - noise.mean()) / noise.std()
    a = a * np.clip(1.0 - np.maximum(0, noise - 2.2) * 3, 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8))
    im = im.rotate(ang, resample=Image.BICUBIC, expand=True)
    im = im.resize((im.width // ss, im.height // ss), Image.BOX)
    a = np.asarray(im, np.float32) / 255
    out = np.full((t.H, W), 15, np.uint8)
    h, w = a.shape
    x0 = int(round(c[0] - w / 2)); y0 = int(round(c[1] - h / 2))
    full = np.zeros((t.H, W), np.float32)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    hh = min(h - sy0, t.H - dy0); ww = min(w - sx0, W - dx0)
    full[dy0:dy0 + hh, dx0:dx0 + ww] = a[sy0:sy0 + hh, sx0:sx0 + ww]
    lvl = np.clip(np.rint(full * 14), 0, 14).astype(np.uint8)
    out[full > 0.08] = np.maximum(lvl[full > 0.08], 1)
    t.px[:, :W] = out
    t.px[:, W:] = 15            # stray texels past the stamp (visible in game)

# ------------------------------------------------------------ SIKEN1 ---

def build_siken1(tl, siken, prev, problems):
    d = bytearray(open('private/unpacked/SIKEN1.BIZ.lz', 'rb').read())
    t = Tim(d, 0x3c800)                  # 512x256 at VRAM (768,0)
    disc = open('private/disc/DAT1/SIKEN.BIN', 'rb').read()
    M = s1_lists(disc)
    keep = [(0, 0, 240, 20), (224, 20, 20, 20)]   # digits, < >, cursor
    free = np.ones(t.px.shape, bool)
    t.px[:] = np.where(free, 0, t.px)
    for x, y, w, h in keep:
        t.px[y:y + h, x:x + w] = t.orig[y:y + h, x:x + w]
        free[y:y + h, x:x + w] = False
    A = Atlas(t, free)
    white, yellow = st_menu(t.rgb(1)), st_yellow(t.rgb(4))
    # menu text pieces are 18 rows (the Japanese used 20) so that 13 rows
    # fit per tpage; drawn at the same y, the text sits 1 px higher
    MH = 18
    def strip(text, style, h=MH, size=15, maxw=None, padto=None):
        a = line_alpha(text, h, size, FONT, maxw=maxw, minsx=0.6)
        if padto:
            l = (padto - a.shape[1]) // 2
            a = pad(a, l, padto - a.shape[1] - l)
        return style(a)
    items = [(0, strip(tl['menu_header'], yellow, h=32, size=26, maxw=250), 1),
             (1, strip(tl['menu_q_exam'], white, maxw=330), len(M[1][1])),
             (8, strip(tl['menu_q_subj'], white, maxw=330), len(M[8][1])),
             (9, strip(tl['menu_kamoku'], white, maxw=60), 1)]
    names = [tl['menu_nyugaku'], tl['menu_exam3'], tl['menu_exam4'], tl['menu_exam5'],
             tl['menu_exam6'], tl['menu_exam7']]
    term = max(int(np.ceil(font(FONT, 15 * 4).getlength(n) / 4)) for n in names[1:5]) + 1
    for k, n in enumerate(names):
        items.append((2 + k, strip(n, white, maxw=250, padto=term if 1 <= k <= 4 else None), 1))
    for L in range(30):
        items.append((16 + L, strip(tl[f'subj{L + 1:02d}'], white, maxw=200), len(M[16 + L][1]) - 2))
    got, widths = A.pack(items, problems, 'SIKEN1')
    lists = {}
    for li, r in got.items():
        h = 32 if li == 0 else MH
        pieces = [s1_piece(x, y, w, h) for x, y, w in r]
        if li >= 16: pieces = [M[li][1][0]] + pieces + [M[li][1][-1]]
        lists[li] = pieces
    for li, pieces in lists.items():
        p, old = M[li]
        write_list(siken, p, pieces, 5, len(old))
    # x immediates: everything centred on x 256
    cx = lambda w: 256 - w // 2
    set_imm(siken, 0x8009ed4c, cx(widths[0])); set_imm(siken, 0x8009ee44, cx(widths[0]))
    set_imm(siken, 0x8009ed70, cx(widths[1])); set_imm(siken, 0x8009ee74, cx(widths[8]))
    for k, ram in enumerate((0x8009ed8c, 0x8009eda4, 0x8009edbc, 0x8009edd4, 0x8009edec, 0x8009ee04)):
        set_imm(siken, ram, cx(widths[2 + k]))
    set_imm(siken, 0x8009eeb8, cx(widths[2]))      # subject screen: entrance
    set_imm(siken, 0x8009eebc, cx(widths[7]))      # year-end
    set_imm(siken, 0x8009eeb0, cx(widths[3]))      # term exams (padded alike)
    longest = max(widths[16 + L] for L in range(30))
    tot = widths[9] + 8 + 20 + longest + 20
    x0 = max(92, 256 - tot // 2)
    set_imm(siken, 0x8009ef58, x0)
    set_imm(siken, 0x8009ef78, x0 + widths[9] + 8)
    if x0 + tot > 422: problems.append(f'SIKEN1: lesson line ends at {x0 + tot} (> 422)')
    t.store()
    prev['SIKEN1'] = (t, None)
    return d, lists

# ------------------------------------------------------------ output ---

def compress_check(name, data, problems):
    slot = os.path.getsize(f'private/disc/DAT2/{name}'); slot += -slot % 2048
    for burst in (0xe00, BURST.get(name, 0xe00)):
        c = lz.compress(bytes(data), burst=burst)
        if len(c) <= slot: break
    out, done = streamsim.run(c)
    peak = burst_peak(c)
    ok = done and out[:len(data)] == bytes(data)
    print(f'{name}: {len(c)} bytes, slot {slot} ({slot - len(c):+d}), burst cap {burst:#x}, '
          f'peak {peak:#x}, streamsim {"ok" if ok else "FAILED"}')
    if len(c) > slot: problems.append(f'{name}: {len(c)} > slot {slot}')
    if not ok: problems.append(f'{name}: streamsim failed')
    return c

def burst_peak(c):
    """Largest output from one 0x800-byte input sector, last 2 sectors excluded."""
    p = burst.bursts(c); last = max(p)
    return max((v for k, v in p.items() if k < last - 1), default=0)

def build(png=False):
    tl = load_tl(); problems = []; prev = {}
    if not os.path.exists('private/build/SIKEN.BIN'):
        print('private/build/SIKEN.BIN missing: run tools/build_exam.py first'); return 1
    siken = bytearray(open('private/build/SIKEN.BIN', 'rb').read())
    gos = {}
    for n in GO_BASE:
        p = f'private/build/{n}' if os.path.exists(f'private/build/{n}') else f'private/disc/DAT1/{n}'
        gos[n] = bytearray(open(p, 'rb').read())
    ce = build_ceft4(tl, siken, prev)
    s2, lists2, clists, hdr = build_siken2(tl, siken, gos, prev, problems)
    s1, lists1 = build_siken1(tl, siken, prev, problems)
    out = {}
    for name, data in (('CEFT4.BIZ', ce), ('SIKEN2.BIZ', s2), ('SIKEN1.BIZ', s1)):
        out[name] = (data, compress_check(name, data, problems))
    if png and problems:
        os.makedirs(PNG, exist_ok=True)
        for k in ('SIKEN2', 'SIKEN1'):
            t = prev[k][0]; before_after(t, 0 if k == 'SIKEN2' else 1, None, k.lower() + '_debug', 2)
    if problems:
        for p in problems: print(p)
        print(f'examgfx: {len(problems)} problems, nothing written'); return 1
    for name, (data, c) in out.items():
        open(f'private/build/{name}.lz', 'wb').write(data)
        open(f'private/build/{name}', 'wb').write(c)
    open('private/build/SIKEN.BIN', 'wb').write(siken)
    for n, g in gos.items(): open(f'private/build/{n}', 'wb').write(g)
    if png: previews(prev, siken, gos['SIKENGO1.BIN'])
    print('examgfx: wrote CEFT4.BIZ SIKEN1.BIZ SIKEN2.BIZ SIKEN.BIN SIKENGO1.BIN SIKENGO2.BIN, problems 0')
    return 0

# ----------------------------------------------------------- previews ---

def tim_rgba(t, k, px=None):
    return t.rgb(k)[t.px if px is None else px]

def save(img, name, scale=2):
    im = Image.fromarray(img.astype(np.uint8))
    if scale != 1: im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(f'{PNG}/{name}.png')

def blit(canvas, src, x, y):
    """Alpha-blit an RGBA array onto an RGB canvas (clipped)."""
    h, w = src.shape[:2]; H, W = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1: return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    m = s[..., 3] > 0
    canvas[y0:y1, x0:x1][m] = s[..., :3][m]

def before_after(t, k, rect=None, name='x', scale=2):
    x, y, w, h = rect or (0, 0, t.W, t.H)
    a = t.rgb(k)[t.orig[y:y + h, x:x + w]]; b = t.rgb(k)[t.px[y:y + h, x:x + w]]
    for im in (a, b): im[im[..., 3] == 0] = (255, 0, 255, 255)
    gap = np.full((h, 6, 4), 255, np.uint8)
    save(np.concatenate([a, gap, b], 1)[..., :3], name, scale)

def previews(prev, siken, go):
    os.makedirs(PNG, exist_ok=True)
    ce, _ = prev['CEFT4']
    before_after(ce, 2, (256, 0, 160, 96), 'ceft4_start_end', 3)
    before_after(ce, 8, (176, 96, 64, 32), 'ceft4_hud', 4)
    s2 = prev['SIKEN2'][0]; pm = prev['SIKEN2pm'][0]; s1 = prev['SIKEN1'][0]
    before_after(s2, 0, None, 'siken2_atlas', 2)
    before_after(pm, 1, None, 'siken2_practice', 3)
    before_after(s1, 1, None, 'siken1_atlas_pal1', 2)
    before_after(s1, 4, (0, 0, 512, 64), 'siken1_atlas_pal4', 2)
    for k in ('stamp_pass', 'stamp_fail'):
        before_after(prev[k][0], 0, None, k, 1)
    lg = prev['logo'][0]
    for k in (0, 3):
        before_after(lg, k, None, f'logo_pal{k}', 4)
    rows = []
    for k in range(7):
        rows.append(np.concatenate([lg.rgb(k)[lg.px][..., :3]] * 3, 1))
    save(np.concatenate(rows, 0), 'logo_tiled', 1)

    # --- HUD and start/end on a grey field (512x240 game screen)
    scr = np.full((240, 512, 3), 90, np.uint8)
    blit(scr, ce.rgb(2)[ce.px[0:96, 256:336]], 216, 72)
    recs = [siken[0x17c0 + 8 * i:0x17c8 + 8 * i] for i in range(3)]
    for tp, clx, cly, u, v, w, h, x in recs:
        blit(scr, ce.rgb(clx)[ce.px[v:v + h, (tp * 64 - 768) * 4 + u:(tp * 64 - 768) * 4 + u + w]], x + 30, 190)
    uv = siken[0x17ae:0x17ae + 2]                # digit "1" cell (tpage 832)
    blit(scr, ce.rgb(8)[ce.px[uv[1]:uv[1] + 16, 256 + uv[0]:256 + uv[0] + 24]], 70, 190)
    save(scr, 'mock_hud_start', 2)
    scr = np.full((240, 512, 3), 90, np.uint8)
    blit(scr, ce.rgb(2)[ce.px[0:96, 336:416]], 216, 72)
    save(scr, 'mock_end', 2)

    # --- results sheet, one per exam
    base = GO_BASE['SIKENGO1.BIN']
    G = s2_lists(go, 0xa7c, 60, base)
    C = s2_lists(go, 0xcfc, 11, base)
    hdr = [struct.unpack_from('<9h', go, 0x12 + 18 * i) for i in range(10)]
    rows = [struct.unpack_from('<H', go, 0xc8 + 2 * i)[0] for i in range(6)]
    tx = struct.unpack_from('<h', go, 0x13d4)[0]
    frame = Tim(s2.buf, 0x1d000)
    def piece_img(e):
        tpx, tpy, cx, cy, u, v, w, h = e
        if tpy == 0:
            x = (tpx * 64 - 768) * 4 + u
            return s2.rgb(0)[s2.px[v:v + h, x:x + w]]
        return pm.rgb(1)[pm.px[v - 128:v - 128 + h, u:u + w]]
    def draw(canvas, pieces, x, y, wrap=False, yref=None):
        for e in pieces:
            img = piece_img(e); w = img.shape[1]
            if wrap and x + w > 480:
                cut = 480 - x
                blit(canvas, img[:, :cut], x, y); x = 120; y += 30; img = img[:, cut:]; w = img.shape[1]
            blit(canvas, img, x, y); x += w
        return x, y
    stamp_p, stamp_f = prev['stamp_pass'][0], prev['stamp_fail'][0]
    for e in range(6):
        scr = np.zeros((480, 512, 3), np.uint8); scr[:] = (90, 10, 10)
        blit(scr, frame.rgb(0)[frame.px], 36, 16)
        for i in range(6):
            tpx, tpy, cly, u, v, w, h, x, y = hdr[i]
            img = s2.rgb(0)[s2.px[v:v + h, (tpx - 768) * 4 + u:(tpx - 768) * 4 + u + w]]
            blit(scr, img, x, y)
        draw(scr, G[49 + e][1], tx, 27)
        for r in range(5):
            draw(scr, G[31 + r][1], 54, rows[r])
            draw(scr, G[e * 5 + r][1], 76, rows[r])
            for c, cx0 in enumerate((240, 299, 358)):
                draw(scr, G[39][1] + G[35][1], cx0 + 10, rows[r])
            draw(scr, G[45][1], 420, rows[r] - 4)
        draw(scr, G[40][1], 64, rows[5])
        img = np.zeros((70, 396, 3), np.uint8); img[:] = 248
        scr[382:452, 100:496] = img
        y = 392
        if e == 0:
            draw(scr, C[5][1], 120, 392, True)
        else:
            x, y = draw(scr, G[e * 5 + (e % 5)][1], 120, 392, True)
            x, y = draw(scr, C[0][1], x, y, True)
            x, y = draw(scr, C[1 + e % 3][1], x, y, True)
            x, y = draw(scr, C[4][1], x, y, True)
            draw(scr, C[5 + e][1], 120, y + 30, True)
        save(scr, f'mock_results_exam{e}', 1)
        if e in (1, 2):
            st = stamp_p if e == 1 else stamp_f
            im = Image.fromarray(st.rgb(0)[st.px]).resize((342, 600), Image.NEAREST)
            blit(scr, np.asarray(im), 85, -80)
            save(scr, f'mock_stamp_{"pass" if e == 1 else "fail"}', 1)
    # practice results menu (positions from SIKEN 0x2c)
    scr = np.zeros((480, 512, 3), np.uint8); scr[:] = (30, 30, 60)
    R = s2_lists(bytes(siken), 0x1310, 60, B)
    for k, li in ((0, 46), (1, 47), (3, 55), (2, 48)):
        x, y = struct.unpack_from('<2H', siken, 0x2c + 4 * k)
        draw(scr, R[li][1], x, y + (32 if k == 2 else 0))
    scr[:, 256] = (255, 0, 0)
    save(scr, 'mock_practice_menu', 1)

    # --- Exam Practice menus
    bg = Tim(prev['SIKEN1'][0].buf, 0)
    M = s1_lists(bytes(siken))
    imm = lambda ram: struct.unpack_from('<h', siken, ram - B)[0]
    def mdraw(canvas, li, x, y, pal=1):
        for tp, u, v, w, h in M[li][1]:
            xx = (tp * 64 - 768) * 4 + u
            blit(canvas, s1.rgb(pal)[s1.px[v:v + h, xx:xx + w]], x, y); x += w
    def screen():
        c = bg.rgb(0)[bg.px][..., :3].copy()
        c[120:376, 86:426] = (c[120:376, 86:426] * 0.35).astype(np.uint8)
        mdraw(c, 0, imm(0x8009ed4c), 48, 4)
        return c
    c = screen()
    mdraw(c, 1, imm(0x8009ed70), 136)
    for k, ram in enumerate((0x8009ed8c, 0x8009eda4, 0x8009edbc, 0x8009edd4, 0x8009edec, 0x8009ee04)):
        mdraw(c, 2 + k, imm(ram), 176 + 32 * k)
    c[:, 256] = (255, 0, 0)
    save(c, 'mock_menu_exam', 1)
    for e in range(6):
        c = screen()
        mdraw(c, 8, imm(0x8009ee74), 136)
        tx = imm(0x8009eeb8) if e == 0 else imm(0x8009eebc) if e == 5 else imm(0x8009eeb0)
        mdraw(c, 2 + e, tx, 176)
        mdraw(c, 15, 0x8e, 240)
        for k in range(5): mdraw(c, 10 + k, 0x8e + 52 * k, 272)
        mdraw(c, 9, imm(0x8009ef58), 336)
        mdraw(c, 16 + e * 5, imm(0x8009ef78), 336)
        c[:, 256] = (255, 0, 0)
        save(c, f'mock_menu_subject{e}', 1)
    # every lesson line, stacked, with the panel edges
    c = np.zeros((30 * 22 + 8, 512, 3), np.uint8); c[:] = (20, 20, 30)
    c[:, 86] = c[:, 426] = (255, 0, 0)
    for L in range(30):
        mdraw(c, 9, imm(0x8009ef58), 4 + 22 * L)
        mdraw(c, 16 + L, imm(0x8009ef78), 4 + 22 * L)
    save(c, 'mock_menu_lessons', 2)
    # results subject names and comment lines, all 30 / all 6
    c = np.full((30 * 26 + 8, 512, 3), 248, np.uint8)
    c[:, 76] = (0, 0, 255); c[:, 228] = (255, 0, 0)
    for L in range(30):
        x, y = draw(c, G[L][1], 76, 4 + 26 * L)
        x, y = draw(c, C[0][1], x + 10, y)
        x, y = draw(c, C[2][1], x, y)
        draw(c, C[4][1], x, y)
    c[:, 480] = (255, 0, 0)
    save(c, 'mock_results_names', 2)
    c = np.full((6 * 26 + 8, 512, 3), 248, np.uint8)
    c[:, 120] = (0, 0, 255); c[:, 480] = (255, 0, 0)
    for k in range(6): draw(c, C[5 + k][1], 120, 4 + 26 * k)
    save(c, 'mock_comments', 2)

if __name__ == '__main__':
    sys.exit(build('--png' in sys.argv))
