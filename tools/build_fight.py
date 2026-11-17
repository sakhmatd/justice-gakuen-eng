# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Fight callouts from the USA arcade disc: DAT2/CEFT1.BIZ (HUD TIM at
0x10800, 4bpp 768x256 at VRAM 768,256) and the exe records that cut it.

Used by the diary Hayato fights (SKHAY) and the arcade modes. The TIM1
pixels and CLUT come from the USA CEFT1 (private/usa/unpacked/CEFT1.BIZ.lz);
the JP and USA files differ only inside TIM1. On top of the USA art the
tool adds:
- big ROUND digits 3-6 in the USA style (the USA has 1 and 2 only), drawn
  in Verdana Bold with the USA outline, shadow and fill texture;
- our fighter names (tl/FIGHT.txt name_*, rotated, Inconsolata Extra
  Condensed Black) in the USA name cells, so they match custom names;
- the JP PUSH BUTTON piece at its JP rect (680,192), which no known record
  uses; the USA moved it.

Exe changes (file offsets; file = RAM - 0x80010000 + 0x800). Inputs:
private/build/SLPS_012.40 (written by hook.py and the earlier tools; only
the bytes below change), private/disc/SLPS_012.40, private/usa/disc/SLUS_006.81.
- 0x77528-0x779bf: message lists 0-17 (USA records), pointer table at
  0x779c0. Lists 7 and 8 keep the JP bytes: the exams draw them from CEFT4.
  Lists 2-5 (ROUND 3-6) are USA list 0 with our digit piece.
- 0x49094/0x49098: win callout x back to the disc value +-184.
- 0x79318: fighter-name table (28 x u8 u, v, h) -> our name strips.
- 11 CEFT1 records moved as on the USA disc (first attack, swap prompts,
  4 banners, READY!!, MAX), see MOVED.
- Hit counter (exe 0x8005e3xx): digit uv table 0x79764 -> USA medium
  digits, and the sizes, CLUTs, label rect (HIT) and P2 x of the USA code,
  see COUNTER.
--png writes previews to private/usa/ceft1/ (callouts as the game places them
on a fight screenshot); --check builds and checks but writes nothing.
Notes: private/usa/ceft1.org.
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import lz, streamsim, burst

EXE = 'private/build/SLPS_012.40'
DISC_EXE = 'private/disc/SLPS_012.40'
USA_EXE = 'private/usa/disc/SLUS_006.81'
SRC_JP = 'private/unpacked/CEFT1.BIZ.lz'
SRC_USA = 'private/usa/unpacked/CEFT1.BIZ.lz'
PNG = 'private/usa/ceft1'
BG = 'private/re/fight/t2/s1200.png'          # diary fight screenshot (512x240)
TL = 'tl/FIGHT.txt'
F_DIGIT = '/usr/share/fonts/TTF/verdanab.ttf'
F_NAME = '/usr/share/fonts/TTF/Inconsolata-ExtraCondensedBlack.ttf'
EB = 0x80010000 - 0x800                   # exe RAM - file offset
TIM1 = 0x10800
LISTS, LISTS_END, PTAB = 0x77528, 0x779c0, 0x779c0
USA_PTAB = 0x71aa4
NAMES, USA_NAMES = 0x79318, 0x73198
KEEP_LISTS = (7, 8)
WIN_X = (0x49094, 0x49098)
# (JP offset, USA offset, halfwords): records whose rect the USA moved.
MOVED = [(0x77072, 0x7142e, 12),            # first attack
         (0x77f2e, 0x72006, 12), (0x77f92, 0x7206a, 12),   # swap prompts P1/P2
         (0x78006, 0x720de, 12),            # Burning Vigor Attack
         (0x78022, 0x720fa, 12),            # Tardy Counter
         (0x7803e, 0x72116, 12),            # Team Up Technique
         (0x7805a, 0x72132, 12),            # FINISH
         (0x79664, 0x734e0, 11), (0x7967a, 0x734f6, 11)]   # READY!! P1/P2
# Non-text records kept JP (user rule: only translated callouts and labels
# come from the USA disc). MAX! P1/P2: same pixels, the USA record is 12
# rows instead of 11. Written from the disc exe (undoes older builds).
KEEP_REC = [(0x796f8, 11), (0x7970e, 11)]
# Non-text CLUT rows (16 colours each) the USA changed; copied back from JP.
KEEP_CLUT = (8, 9, 10, 11, 34)
# Hit counter: digit uv table (u8 u, v per digit 0-9) and code immediates
# (RAM, JP imm, USA imm), taken from the USA function 0x8005a1d8.
DIGIT_TAB, USA_DIGIT_TAB = 0x79764, 0x735d8
COUNTER = [(0x8005e40c, 0x178, 0x158),    # P2 x
           (0x8005e48c, 0x10, 0x30),      # tens: CLUT x (16,502) -> (48,502)
           (0x8005e4ac, 0x36, 0x3e),      # tens: y
           (0x8005e4b0, 0x1f, 0x20),      # tens: w
           (0x8005e4b8, 0x17, 0x10),      # tens: h
           (0x8005e4d8, 0x1f, 0x20),      # tens: uv w
           (0x8005e4e4, 0x0f, 0x10),      # tens: uv h
           (0x8005e5b0, 0x10, 0x30),      # units: CLUT
           (0x8005e5d4, 0xfff8, 0x0000),  # units: y = 0x3e - 8 -> 0x3e
           (0x8005e5d8, 0x1f, 0x20),
           (0x8005e5e0, 0x17, 0x10),
           (0x8005e600, 0x1f, 0x20),
           (0x8005e60c, 0x0f, 0x10),
           (0x8005e66c, 0x10, 0x20),      # label: CLUT (16,502) -> (32,502)
           (0x8005e684, 0x3e, 0x42),      # label: y
           (0x8005e688, 0x20, 0x40),      # label: w
           (0x8005e68c, 0x10, 0x0c),      # label: h (quad and uv)
           (0x8005e6a0, 0x50, 0x80),      # label: u   (発 -> HIT)
           (0x8005e6a4, 0xc0, 0xd0),      # label: v
           (0x8005e6a8, 0x20, 0x40)]      # label: uv w
# Exe byte ranges this tool writes (for RAM pokes in the harness).
RANGES = ([(LISTS, PTAB + 72), (NAMES, NAMES + 84), (WIN_X[0], WIN_X[1] + 4),
           (DIGIT_TAB, DIGIT_TAB + 20)] + [(j, j + 2 * n) for j, _, n in MOVED]
          + [(j, j + 2 * n) for j, n in KEEP_REC]
          + [(r - EB, r - EB + 4) for r, _, _ in COUNTER])

# ---------------------------------------------------------------- TIM ---

class Tim:
    """4bpp TIM inside a decompressed buffer."""
    def __init__(self, buf, off):
        self.buf = buf
        bl = struct.unpack_from('<I', buf, off + 8)[0]
        cw, ch = struct.unpack_from('<2H', buf, off + 16)
        self.clut = struct.unpack_from(f'<{cw * ch}H', buf, off + 20)
        p = off + 8 + bl
        _, vx, vy, w, h = struct.unpack_from('<IHHHH', buf, p)
        self.pix, self.stride, self.H = p + 12, w * 2, h
        raw = np.frombuffer(bytes(buf[self.pix:self.pix + w * 2 * h]), np.uint8).reshape(h, w * 2)
        self.px = np.empty((h, w * 4), np.uint8)
        self.px[:, 0::2] = raw & 15; self.px[:, 1::2] = raw >> 4
        self.W = self.px.shape[1]
        self.orig = self.px.copy()

    def store(self):
        raw = (self.px[:, 0::2] | (self.px[:, 1::2] << 4)).astype(np.uint8)
        self.buf[self.pix:self.pix + self.stride * self.H] = raw.tobytes()

    def rgb(self, k):
        """(16, 4) RGBA; CLUT value 0 is transparent."""
        return np.array([((v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3, 255 if v else 0)
                         for v in self.clut[k * 16:(k + 1) * 16]], np.uint8)

# ---------------------------------------------------------- rendering ---

_fonts = {}
def font(path, size):
    if (path, size) not in _fonts: _fonts[path, size] = ImageFont.truetype(path, size)
    return _fonts[path, size]

def text_alpha(text, cap, path, maxw=None, minsx=0.6, ss=4, sx=1.0, track=0):
    """Alpha of one line with capitals `cap` px tall, rows = cap + 2 (one
    spare row above and below). Squeezed horizontally to maxw (not below
    minsx, then the size drops)."""
    while True:
        size = 8
        while True:                          # font size whose 'H' is cap px
            f = font(path, size * ss)
            b = f.getbbox('H', anchor='ls')
            if (b[3] - b[1]) / ss >= cap - 0.25: break
            size += 1
        adv = f.getlength(text) / ss + track * (len(text) - 1)
        s = sx
        if maxw and adv * s > maxw: s = maxw / adv
        if s >= minsx or cap <= 8: break
        cap -= 1
    h = cap + 2
    W = int(np.ceil(adv)) + 4
    im = Image.new('L', (W * ss, h * ss), 0)
    dr = ImageDraw.Draw(im)
    for i, ch in enumerate(text):
        x = 2 * ss + f.getlength(text[:i]) + track * ss * i
        dr.text((x, (1 + cap) * ss), ch, font=f, fill=255, anchor='ls')
    w = max(1, int(round(W * s)))
    a = np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255
    cols = np.nonzero(a.max(0) > 0.05)[0]
    return a[:, cols[0]:cols[-1] + 1] if len(cols) else a

def shift(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out

DISK2 = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], bool)

# ------------------------------------------------------ ROUND digits ---

# The USA big digits 1 and 2 are 48x48 cells at (256,144) and (304,144),
# CLUT 60: fill 9-13 (mottled texture), 2 px outline 1 (2 on the top
# edge), shadow 8 one or two px to the lower right. Fill box of the '2':
# x 2-43, y 7-41. New cells for 3-5 in free grid texels (tpage 896).
# No free 48x48 is left for 6 (the swap-face cells 640-767,208-239 are
# filled at runtime), so 6 is cut into three 48x16 strips in the free row
# (0-143, 96-111) of tpage 768, drawn as three records stacked with the
# piece offset a (-16, 0, 16) so they zoom as one.
DIGIT_CELLS = {3: (608, 144), 4: (656, 144), 5: (704, 144)}
DIGIT_STRIPS = {6: [(0, 96), (48, 96), (96, 96)]}
DIGIT_H, DIGIT_TOP, DIGIT_W2 = 35, 7, 42

def digit_mask(d):
    f = font(F_DIGIT, 160)
    im = Image.new('L', (200, 220), 0)
    ImageDraw.Draw(im).text((20, 10), str(d), font=f, fill=255)
    a = np.asarray(im, np.float32) / 255
    ys, xs = np.nonzero(a > 0.5)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def big_digit(d, tex, texm):
    """48x48 cell: digit d in the USA ROUND-digit style. tex/texm: the USA
    ROUND piece (48 rows) and its fill mask, sampled row by row."""
    m2 = digit_mask(2)
    k = DIGIT_W2 / (m2.shape[1] / m2.shape[0] * DIGIT_H)    # widen like the USA '2'
    m = digit_mask(d)
    h, w = m.shape
    W = min(DIGIT_W2, int(round(w / h * DIGIT_H * k)))
    s = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((W, DIGIT_H), Image.BOX),
                   np.float32) / 255 > 0.5
    F = np.zeros((48, 48), bool)
    x0 = 23 - W // 2
    F[DIGIT_TOP:DIGIT_TOP + DIGIT_H, x0:x0 + W] = s
    S = ndimage.binary_dilation(F, DISK2)
    out = np.zeros((48, 48), np.uint8)
    out[(shift(S, 1, 1) | shift(S, 2, 2) | shift(S, 1, 2) | shift(S, 2, 1)) & ~S] = 8
    out[S] = 1
    out[S & ~shift(S, 1, 0)] = 2
    for r in range(48):
        cols = np.nonzero(F[r])[0]
        if not len(cols): continue
        seq = tex[r][texm[r]]
        out[r, cols] = seq[(40 * d + cols - cols[0]) % len(seq)]
    return out

def draw_digits(t):
    tex = t.px[0:48, 256:512]
    texm = tex >= 9
    for d, (x, y) in DIGIT_CELLS.items():
        t.px[y:y + 48, x:x + 48] = big_digit(d, tex, texm)
    for d, cells in DIGIT_STRIPS.items():
        c = big_digit(d, tex, texm)
        for k, (x, y) in enumerate(cells):
            t.px[y:y + 16, x:x + 48] = c[16 * k:16 * k + 16]

# ------------------------------------------------------------- lists ---

def read_lists(exe, ptab):
    L = {}
    for i in range(18):
        p = struct.unpack_from('<I', exe, ptab + 4 * i)[0] - EB
        n = struct.unpack_from('<h', exe, p)[0]
        L[i] = [struct.unpack_from('<15h', exe, p + 2 + 30 * k) for k in range(n)]
    return L

def build_lists(usa):
    U = read_lists(usa, USA_PTAB)
    L = {i: U[i] for i in (0, 1, 6, 10, 11, 12, 13, 14, 15, 16, 17)}
    word, dig = U[0]
    for n in range(3, 7):
        if n in DIGIT_CELLS:
            x, y = DIGIT_CELLS[n]
            r = list(dig)
            r[3] = 768 + 64 * (x // 256); r[7] = x % 256; r[8] = y
            L[n - 1] = [word, tuple(r)]
            continue
        recs = [word]
        for k, (x, y) in enumerate(DIGIT_STRIPS[n]):
            r = list(dig)
            r[1] = dig[1] - 16 + 16 * k
            r[3] = 768 + 64 * (x // 256); r[7] = x % 256; r[8] = y
            r[10] = 16; r[14] = 16
            recs.append(tuple(r))
        L[n - 1] = recs
    L[9] = L[0]
    return L

def pack_lists(exe, disc, L):
    """Rewrite the list block and the pointer table. Lists 7, 8 keep the
    disc bytes."""
    ptr = lambda i: struct.unpack_from('<I', disc, PTAB + 4 * i)[0] - EB
    blob = bytearray(); where = {}
    for i in range(18):
        if i == 9: continue
        if i in KEEP_LISTS:
            p = ptr(i); n = struct.unpack_from('<h', disc, p)[0]
            body = disc[p:p + 2 + 30 * n]
        else:
            body = struct.pack('<h', len(L[i])) + b''.join(struct.pack('<15h', *r) for r in L[i])
        where[i] = LISTS + len(blob); blob += body
    where[9] = where[0]
    if LISTS + len(blob) > LISTS_END:
        return f'lists need {len(blob)} bytes, block has {LISTS_END - LISTS}'
    blob += bytes(LISTS_END - LISTS - len(blob))
    exe[LISTS:LISTS_END] = blob
    for i in range(18):
        struct.pack_into('<I', exe, PTAB + 4 * i, where[i] + EB)
    return None

def patch_records(exe, disc, usa, problems):
    for o in WIN_X:
        exe[o:o + 4] = disc[o:o + 4]
    for j, u, n in MOVED:
        a = struct.unpack_from(f'<{n}h', disc, j); b = struct.unpack_from(f'<{n}h', usa, u)
        if a[1] != 256 or b[1] != 256 or a[3] != b[3]:
            problems.append(f'record {j:#x}: JP/USA pairing looks wrong'); continue
        exe[j:j + 2 * n] = usa[u:u + 2 * n]
    exe[DIGIT_TAB:DIGIT_TAB + 20] = usa[USA_DIGIT_TAB:USA_DIGIT_TAB + 20]
    for ram, old, new in COUNTER:
        o = ram - EB
        w = struct.unpack_from('<I', disc, o)[0]
        if w & 0xffff != old:
            problems.append(f'counter: unexpected code at {ram:#x} ({w:#010x})'); continue
        struct.pack_into('<I', exe, o, (w & 0xffff0000) | new)

# -------------------------------------------------------------- names ---

# JP name cell (u, v) -> tl key (JP name table, same character ids).
NAME_KEYS = {(0, 168): 'kyosuke', (24, 168): 'batsu', (48, 168): 'hinata', (72, 168): 'shoma',
             (96, 168): 'natsu', (120, 168): 'roberto', (144, 168): 'roy', (208, 112): 'tiffany',
             (24, 200): 'boman', (48, 200): 'edge', (72, 200): 'akira', (96, 200): 'gan',
             (120, 200): 'hideo', (144, 200): 'kyoko', (168, 200): 'raizo', (192, 200): 'hyo',
             (216, 200): 'sakura', (232, 80): 'daigo', (232, 112): 'hayato'}
NAME_CAP = 12
NAME_ADV = 5.8

def name_style(a):
    """Vertical name (CLUT 1): yellow 5, darker 7 on the right edge of the
    strokes, black a ring and a shadow to the lower right."""
    m = a > 0.45
    k = ndimage.binary_dilation(m, np.ones((3, 3), bool))
    out = np.zeros(a.shape, np.uint8)
    out[k | shift(k, 1, 1)] = 10
    out[m] = 5
    out[m & ~shift(m, 0, -1)] = 7
    return out

def name_strip(text):
    """24 x len strip: the name rotated to read top to bottom."""
    a = text_alpha(text, NAME_CAP, F_NAME, maxw=NAME_ADV * len(text), minsx=0.3)
    h, w = a.shape
    full = np.zeros((22, w + 3), np.float32)
    y = (22 - h) // 2
    full[y:y + h, 1:1 + w] = a
    r = np.rot90(full, -1)                    # clockwise: top of letters to the right
    s = np.zeros((r.shape[0], 24), np.float32)
    s[:, 0:22] = r
    return name_style(s)

def name_columns(usa):
    """The USA name cells (ids 0-26), merged into columns {u: [v0, v1)}."""
    cells = {tuple(usa[USA_NAMES + 3 * i:USA_NAMES + 3 * i + 3]) for i in range(27)}
    cols = {}
    for u, v, h in sorted(cells):
        cols.setdefault(u, []).append((v, v + h))
    out = []
    for u, rs in cols.items():
        rs.sort(); cur = list(rs[0])
        for a, b in rs[1:]:
            if a <= cur[1]: cur[1] = max(cur[1], b)
            else: out.append((u, *cur)); cur = [a, b]
        out.append((u, *cur))
    return cells, out

def build_names(tl, t, tj, exe, disc, usa, problems):
    cells, cols = name_columns(usa)
    for u, v, h in cells:
        t.px[v:v + h, u:u + 24] = 0
    for x, y, w, h in KEEP_JP:                # JP pieces the JP code needs
        t.px[y:y + h, x:x + w] = tj.px[y:y + h, x:x + w]
        new = []
        for u, v0, v1 in cols:
            if u < x + w and x < u + 24 and v0 < y + h and y < v1:
                if v0 >= y: v0 = y + h            # piece at the column top
                else: v1 = y                      # piece below the top
                if v0 >= v1: continue
            new.append((u, v0, v1))
        cols = new
    cols += EXTRA_NAME_COLS
    strips = {k: name_strip(tl['name_' + k]) for k in NAME_KEYS.values()}
    room = {(u, v0): v0 for u, v0, v1 in cols}
    end = {(u, v0): v1 for u, v0, v1 in cols}
    slot = {}
    for k in sorted(strips, key=lambda k: -strips[k].shape[0]):
        h = strips[k].shape[0]
        fit = [c for c in room if end[c] - room[c] >= h]
        if not fit:
            problems.append(f'name {k}: no column has {h} px left'); continue
        c = min(fit, key=lambda c: end[c] - room[c])         # best fit
        v = room[c]
        t.px[v:v + h, c[0]:c[0] + 24] = strips[k]
        slot[k] = (c[0], v, h); room[c] = v + h
    for i in range(28):
        u, v, h = disc[NAMES + 3 * i:NAMES + 3 * i + 3]
        k = NAME_KEYS.get((u, v))
        if k is None or k not in slot:
            exe[NAMES + 3 * i:NAMES + 3 * i + 3] = disc[NAMES + 3 * i:NAMES + 3 * i + 3]
            continue
        exe[NAMES + 3 * i:NAMES + 3 * i + 3] = bytes(slot[k])
    return slot

# ------------------------------------------------------------- output ---

def load_tl(path=TL):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'): continue
        k, v = line.split(':', 1)
        out[k.strip()] = v.strip()
    return out

def compress_check(data, problems):
    slot = os.path.getsize('private/disc/DAT2/CEFT1.BIZ'); slot += -slot % 2048
    for cap in (0xe00, 0x1200, 0x1600):
        c = lz.compress(bytes(data), burst=cap)
        if len(c) <= slot: break
    out, done = streamsim.run(c)
    p = burst.bursts(c); last = max(p)
    peak = max((v for k, v in p.items() if k < last - 1), default=0)
    ok = done and out[:len(data)] == bytes(data)
    print(f'CEFT1.BIZ: {len(c)} bytes, slot {slot} ({slot - len(c):+d}), burst cap {cap:#x}, '
          f'peak {peak:#x}, streamsim {"ok" if ok else "FAILED"}')
    if len(c) > slot: problems.append(f'CEFT1.BIZ: {len(c)} > slot {slot}')
    if not ok: problems.append('CEFT1.BIZ: streamsim failed')
    return c

PUSH = (680, 192, 88, 16)
# JP pieces drawn by JP code with runtime uv, restored over the USA name
# cells: school emblems 24x16 at (0-119, 240) and the triangle at (0,224)
# (HUD records 0x79414/0x7942a, tpage 768; the USA moved them to tpage
# 832 with other uv code).
# The 1x8 texel column (0,168) is the source of HUD record 0x795c0 (w 0,
# set at runtime); kept JP too, and the name column starts below it.
KEEP_JP = [(0, 224, 24, 16), (0, 240, 120, 16), (0, 168, 1, 8)]
# Extra name columns (u, v0, v1) in grid texels that are unused on both
# discs (tpage 768, x 176-223, y 80-111).
EXTRA_NAME_COLS = [(176, 80, 112), (200, 80, 112)]

def build(png=False, check=False):
    tl = load_tl(); problems = []
    if not os.path.exists(EXE):
        print(f'{EXE} missing: run tools/hook.py first'); return 1
    exe = bytearray(open(EXE, 'rb').read())
    disc = open(DISC_EXE, 'rb').read()
    usa = open(USA_EXE, 'rb').read()
    if len(exe) != len(disc): problems.append('exe size differs from disc')
    jp = open(SRC_JP, 'rb').read()
    d = bytearray(open(SRC_USA, 'rb').read())
    if len(d) != len(jp) or d[:TIM1] != jp[:TIM1]:
        problems.append('USA CEFT1 differs from JP outside TIM1')
    t = Tim(d, TIM1)
    tj = Tim(bytearray(jp), TIM1)
    free = t.px[PUSH[1]:PUSH[1] + PUSH[3], PUSH[0]:PUSH[0] + PUSH[2]]
    if len(set(free.ravel().tolist()) - {0, 7}) > 0:
        problems.append('PUSH BUTTON rect is not free in the USA atlas')
    t.px[PUSH[1]:PUSH[1] + PUSH[3], PUSH[0]:PUSH[0] + PUSH[2]] = \
        tj.px[PUSH[1]:PUSH[1] + PUSH[3], PUSH[0]:PUSH[0] + PUSH[2]]
    cells = [(x, y, 48, 48) for x, y in DIGIT_CELLS.values()]
    cells += [(x, y, 48, 16) for v in DIGIT_STRIPS.values() for x, y in v]
    for x, y, w, h in cells:
        if len(set(t.px[y:y + h, x:x + w].ravel().tolist()) - {0, 7}) > 0:
            problems.append(f'digit cell {x},{y} is not free')
    for k in KEEP_CLUT:                       # non-text palettes back to JP
        o = TIM1 + 20 + 32 * k
        d[o:o + 32] = jp[o:o + 32]
    t.clut = struct.unpack_from(f'<{len(t.clut)}H', d, TIM1 + 20)
    draw_digits(t)
    L = build_lists(usa)
    err = pack_lists(exe, disc, L)
    if err: problems.append(err)
    patch_records(exe, disc, usa, problems)
    for j, n in KEEP_REC:
        exe[j:j + 2 * n] = disc[j:j + 2 * n]
    slots = build_names(tl, t, tj, exe, disc, usa, problems)
    t.store()
    c = compress_check(d, problems)
    if png: previews(t, exe, slots)
    if problems:
        for p in problems: print(p)
        print(f'fight: {len(problems)} problems, nothing written'); return 1
    if check:
        print('fight: --check, problems 0, nothing written'); return 0
    open('private/build/CEFT1.BIZ.lz', 'wb').write(d)
    open('private/build/CEFT1.BIZ', 'wb').write(c)
    open(EXE, 'wb').write(exe)
    print('fight: wrote CEFT1.BIZ (USA art), SLPS_012.40 (lists, records, names, counter), problems 0')
    return 0

# ----------------------------------------------------------- previews ---

def piece(t, tpx, clx, cly, u, v, w, h):
    x = (tpx - 768) * 4 + u
    c = (cly - 500) * 32 + clx // 16
    return t.rgb(c)[t.px[v:v + h, x:x + w]]

def blit(canvas, src, x, y):
    h, w = src.shape[:2]; H, W = canvas.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1: return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    m = s[..., 3] > 0
    canvas[y0:y1, x0:x1][m] = s[..., :3][m]

def scaled(img, w, h):
    return np.asarray(Image.fromarray(img).resize((max(1, w), max(1, h)), Image.NEAREST))

def list_screen(t, recs, bg, cx_list, cy0=105):
    """Final (zoomed-in) frame of one message list. Quad = sw+1 x sh+1
    centred on x, y = cy0 + a + b (builder 0x80058c98)."""
    scr = bg.copy()
    for cx in cx_list:
        for r in recs:
            tpx, tpy, clx, cly, u, v, tw, th = r[3:11]
            img = piece(t, tpx, clx, cly, u, v, max(tw, 1), max(th, 1))
            hw, hh = (r[13] + 1) // 2, (r[14] + 1) // 2
            img = scaled(img, 2 * hw, 2 * hh)
            blit(scr, img, cx - hw, cy0 + r[1] + r[2] - hh)
    return scr

def previews(t, exe, slots):
    os.makedirs(PNG, exist_ok=True)
    bg = np.asarray(Image.open(BG).convert('RGB')).copy()
    L = read_lists(exe, PTAB)
    names = {0: 'ROUND 1', 1: 'ROUND 2', 2: 'ROUND 3', 3: 'ROUND 4', 4: 'ROUND 5', 5: 'ROUND 6',
             6: 'FINAL ROUND', 10: 'FIGHT', 11: 'K.O.', 12: 'DRAW', 13: 'TIME OVER',
             14: 'win: perfect', 15: 'win', 16: 'win: close', 17: 'CPU wins'}
    tiles = []
    for i, lab in names.items():
        win = i >= 14
        scr = list_screen(t, L[i], bg, (256 - 184, 256 + 184) if win else (256,))
        im = Image.fromarray(scr); ImageDraw.Draw(im).text((4, 226), f'list {i}: {lab}', fill=(255, 255, 255))
        tiles.append(np.asarray(im))
    for k in range(0, len(tiles), 6):
        row = tiles[k:k + 6]
        while len(row) < 6: row.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(row[0:3]), np.hstack(row[3:6])])
        Image.fromarray(sheet).save(f'{PNG}/prev_lists{k // 6}.png')
    # HUD pieces at their screen positions: banners, first attack, swap
    # prompts, READY, MAX, counter, PUSH BUTTON, names.
    scr = bg.copy()
    def rec(o, n):
        return struct.unpack_from(f'<{n}h', exe, o)
    for o, y in ((0x78006, 40), (0x78022, 70), (0x7803e, 90), (0x7805a, 120)):
        r = rec(o, 12)
        img = scaled(piece(t, r[0], r[2], r[3], r[4], r[5], r[6], r[7]), r[10], r[11])
        blit(scr, img, 380, y)
    r = rec(0x77072, 12)
    blit(scr, scaled(piece(t, r[0], r[2], r[3], r[4], r[5], r[6], r[7]), r[10], r[11]), 40, 48)
    for o in (0x77f2e,):
        r = rec(o, 12)
        blit(scr, piece(t, r[0], r[2], r[3], r[4], r[5], r[6], r[7]), r[8], r[9] - 40)
        blit(scr, piece(t, r[0], r[2], r[3], r[4], r[5] + 16, r[6], r[7]), r[8], r[9] - 26)
    for o in (0x79664, 0x796f8):
        r = rec(o, 11)
        blit(scr, piece(t, r[0], r[2], r[3], r[4], r[5], r[6], r[7]), r[8], r[9])
    tab = exe[DIGIT_TAB:DIGIT_TAB + 20]
    x = 0x28
    for dgt in (1, 2):
        u, v = tab[2 * dgt], tab[2 * dgt + 1]
        blit(scr, piece(t, 832, 48, 502, u, v, 32, 16), x, 0x3e); x += 32
    blit(scr, piece(t, 832, 32, 502, 0x80, 0xd0, 64, 12), x, 0x42)
    blit(scr, piece(t, 896, 240, 501, 168, 192, 88, 16), 210, 150)
    for k, xx in (('kyosuke', 8), ('hayato', 478)):
        u, v, h = slots[k]
        blit(scr, t.rgb(1)[t.px[v:v + h, u:u + 24]], xx, 48)
    Image.fromarray(scr).save(f'{PNG}/prev_hud.png')
    # names side by side, digits 1-6 at 2x
    scr = np.zeros((70, 19 * 26 + 4, 3), np.uint8); scr[:] = (60, 50, 70)
    for j, (k, (u, v, h)) in enumerate(sorted(slots.items())):
        blit(scr, t.rgb(1)[t.px[v:v + h, u:u + 24]], 4 + 26 * j, 4)
    Image.fromarray(scr).save(f'{PNG}/prev_names.png')
    cells = [(256, 144), (304, 144)] + [DIGIT_CELLS[d] for d in range(3, 6)]
    scr = np.zeros((52, 6 * 52, 3), np.uint8); scr[:] = (40, 40, 90)
    for j, (x, y) in enumerate(cells):
        blit(scr, t.rgb(60)[t.px[y:y + 48, x:x + 48]], 2 + 52 * j, 2)
    for k, (x, y) in enumerate(DIGIT_STRIPS[6]):
        blit(scr, t.rgb(60)[t.px[y:y + 16, x:x + 48]], 2 + 52 * 5, 2 + 16 * k)
    Image.fromarray(scr).resize((scr.shape[1] * 2, scr.shape[0] * 2), Image.NEAREST).save(f'{PNG}/prev_digits.png')

if __name__ == '__main__':
    sys.exit(build('--png' in sys.argv, '--check' in sys.argv))
