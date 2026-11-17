# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Diary-mode image text, phase 1 (private/plan-graphics.org): hub date and
weekday, club results banner, Continue screen school names.

Reads tl/DIARYGFX.txt. Run after build_menus.py and build_logo.py:
it edits their outputs in private/build/ (MA_COCK.BIZ, MA_TIT.BIZ,
MZPANEL.BIZ, MGAME.BIN, MGFIN.BIN) and refuses to run twice on the same
MGAME.BIN. --png writes previews to private/re/phase1/png/, --dry writes nothing else.

Date header (MGAME layout 1 = hub, layout 2 = calendar screen at x+96,
MGFIN layout 1 = year-end). The pieces come from MA_COCK TIM1 tpage 512,
palette (176,500). Layout 2 sets tpage 512 for its digit and weekday
sprites in code, so the weekday pieces must stay in tpage 512.
- 月 becomes "/", 日 and the parentheses are blanked (1x1 transparent
  texel). The kanji cells 月火水木金土日 and the parentheses are freed and
  hold the seven weekday pieces (WD_SLOTS); "/" goes to free texels at
  (168,216).
- Digit x tables move the day left. The digit loops skip the day's tens
  digit when it is 0; a code patch then draws the ones digit in the tens
  slot, so "4/8" has no gap. Layout 1 code is rewritten in place (same
  code in MGAME and MGFIN); the new layout 2 day-ones case lives in the
  space this frees and is reached through the switch's jump table.
- MA_COCK tx 0-79 v96-119 (ジャスティス学園) is shown by the school-name
  records at MGAME 0x3554 / MGFIN 0x1c70; it is redrawn as Justice High in
  the build_menus.py style, and the leftover kanji rows under the other
  school labels in those 80x24 cells are cleared.

Club results banner: MZPANEL entry 0, 8bpp TIM at 0x46800, 256x104 = two
256x52 strips drawn side by side as one 512x52 line (visible x 16-495).
The text is painted out and redrawn with build_banners.py methods.

Continue screen: MA_TIT TIM0 (4bpp 80x120, palette 5), five 80x24 labels
at v 0 Taiyo, 24 Gorin, 48 Gedo, 72 Pacific, 96 Justice. The 太陽学園
plaque on the Enroll/Continue photo (MA_TIT 0x3800) becomes TAIYO HIGH.

Calendar screen (private/re/calendar/notes.org): the event label TIM in
MZPANEL entries 0-2 (0x1e800) is redrawn as 15 horizontal 116x40 boxes
with a 122 px cursor, and the MGAME label, event and cursor tables are
rewritten. The month headers lose the 月 glyph (the number is centred).
MZPANEL entries 0-2 also get the logo tile (build_logotile.patch).
"""
import sys, os, struct, collections
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import lz
from hook import Asm, R

OUT = 'private/re/phase1/png'
DRY = False                             # --dry: previews only, write nothing

def write(path, data):
    if not DRY: open(path, 'wb').write(data)
FONT_DATE = '/usr/share/fonts/TTF/timesi.ttf'
B = 0x8009c000

def load_tl(path='tl/DIARYGFX.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'): continue
        k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

def rgb(c): return np.array([(c & 31) * 8, (c >> 5 & 31) * 8, (c >> 10 & 31) * 8], np.uint8)

def unpack4(d, off, w, h):
    raw = np.frombuffer(bytes(d[off:off + w * h // 2]), np.uint8).reshape(h, w // 2)
    px = np.empty((h, w), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return px

def pack4(px): return (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()

def save_png(name, idx, pal, scale=3, bg=(0, 0, 96)):
    os.makedirs(OUT, exist_ok=True)
    im = np.array([rgb(c) for c in pal])[idx]
    im[idx == 0] = bg
    Image.fromarray(im).resize((idx.shape[1] * scale, idx.shape[0] * scale), Image.NEAREST).save(
        f'{OUT}/{name}.png')

# ---------------------------------------------------------------- date
# Weekday slots in MA_COCK tpage 512 (u, v, w, h); 22-row slots hold the
# letters with the baseline at row 20 like the digits (rows 2-21).
WD_SLOTS = {'wd5': (48, 168, 34, 22), 'wd6': (82, 168, 34, 22),   # was 水 土 日
            'wd1': (48, 190, 48, 22), 'wd3': (48, 212, 48, 22),   # was 木 ( 金 )
            'wd4': (48, 234, 48, 22),                             # + free v240-255
            'wd0': (48, 120, 32, 24), 'wd2': (48, 144, 32, 24)}   # was 月 火 (+ free)
FREED = [(48, 120, 32, 48), (48, 168, 72, 24), (48, 192, 48, 64)]
SLASH = (168, 216, 16, 24)
BLANK = (0, 163)                       # a transparent texel for hidden pieces
DATE_PAL = 11                          # CLUT (176,500)
# Layout 1 x positions: month tens/ones, "/", day tens/ones, weekday.
# The slash and day sit 4 and 5 px right of the original grid, which leaves
# at least 4 px between the italic slash and every month or day digit.
# The weekday's left ink edge is at x 131, the club label's left edge (task
# B4). Layout 2 (calendar screen header, x + 96) keeps its weekday at the
# old 106 + 96 = 202 (X2_WD).
X1 = dict(m10=21, m1=35, slash=55, d10=64, d1=78, wd=130)
X2_WD = 202

def date_mask(text, h, cap, ss=4, maxw=None):
    """Coverage (h, w) of text in the date font, baseline at row 20."""
    f = ImageFont.truetype(FONT_DATE, 8)
    size = 8
    while True:
        f = ImageFont.truetype(FONT_DATE, size * ss)
        b = f.getbbox('H')
        if b[3] - b[1] >= cap * ss: break
        size += 1
    asc = f.getmetrics()[0]
    bb = f.getbbox(text)
    W = bb[2] + 2 * ss; H = h * ss
    im = Image.new('L', (W, H), 0)
    base = 20 * ss
    ImageDraw.Draw(im).text((ss, base - asc), text, font=f, fill=255)
    a = np.asarray(im, float)
    xs = np.where(a.max(0) > 0)[0]
    a = a[:, xs[0]:xs[-1] + 1]
    w = (a.shape[1] + ss - 1) // ss
    if maxw and w > maxw: w = maxw
    small = Image.fromarray(a.astype(np.uint8)).resize((w, h), Image.LANCZOS)
    return np.asarray(small, float) / 255

def date_piece(text, w, h, cap, align='left', pad=1):
    """Index image in the style of the date glyphs: white core, grey-blue
    anti-aliased edges (palette 11 is a dark-to-white ramp) and a dark
    1 px shadow to the lower right."""
    a = date_mask(text, h, cap, maxw=w - 1 - pad)
    out = np.zeros((h, w), np.uint8)
    x0 = pad if align == 'left' else (w - 1 - a.shape[1]) // 2
    A = np.zeros((h, w)); A[:, x0:x0 + a.shape[1]] = a
    sh = np.zeros_like(A); sh[1:, 1:] = A[:-1, :-1]
    out[(sh > 0.3) & (A < 0.25)] = 2
    out[(sh > 0.6) & (A < 0.25)] = 3
    m = A >= 0.25
    out[m] = np.clip(np.rint(5 + A[m] * 10.4), 6, 15).astype(np.uint8)
    return out

def build_date_gfx(tl, px, clut, png):
    for x, y, w, h in FREED: px[y:y + h, x:x + w] = 0
    sx, sy, sw, sh = SLASH
    assert not px[sy:sy + sh, sx:sx + sw].any(), 'slash slot not free'
    assert px[BLANK[1], BLANK[0]] == 0
    slash = date_piece(tl.get('date_month', '/') or '/', sw, sh, 18, align='centre')
    px[sy:sy + sh, sx:sx + sw] = slash
    pieces = {}
    for k, (u, v, w, h) in WD_SLOTS.items():
        img = date_piece(tl[k], w, h, 15)
        used = np.where(img.any(0))[0]
        px[v:v + h, u:u + w] = img
        pieces[k] = (u, v, int(used[-1]) + 1, h)
        if png: save_png(f'date_{k}', img, clut[DATE_PAL * 16:DATE_PAL * 16 + 16])
    if png:
        save_png('date_slash', slash, clut[DATE_PAL * 16:DATE_PAL * 16 + 16])
        # mock-up of the header line "4/18 Wed" and "12/8 Sun"
        for name, (m, d, wd) in {'mock_0418': (4, 18, 'wd3'), 'mock_1208': (12, 8, 'wd0')}.items():
            row = np.zeros((26, 190), np.uint8)
            dig = {0: (168, 192), 1: (96, 192), 2: (96, 216), 3: (120, 168), 4: (120, 192),
                   5: (120, 216), 6: (144, 168), 7: (144, 192), 8: (144, 216), 9: (168, 168)}
            def put(x, img):
                h, w = img.shape; sub = row[0:h, x:x + w]
                sub[img > 0] = img[img > 0]
            if m >= 10: put(X1['m10'], px[dig[m // 10][1]:dig[m // 10][1] + 24, dig[m // 10][0]:dig[m // 10][0] + 24])
            put(X1['m1'], px[dig[m % 10][1]:dig[m % 10][1] + 24, dig[m % 10][0]:dig[m % 10][0] + 24])
            put(X1['slash'], slash)
            if d >= 10:
                put(X1['d10'], px[dig[d // 10][1]:dig[d // 10][1] + 24, dig[d // 10][0]:dig[d // 10][0] + 24])
                put(X1['d1'], px[dig[d % 10][1]:dig[d % 10][1] + 24, dig[d % 10][0]:dig[d % 10][0] + 24])
            else:
                put(X1['d10'], px[dig[d][1]:dig[d][1] + 24, dig[d][0]:dig[d][0] + 24])
            u, v, w, h = pieces[wd]
            put(X1['wd'], px[v:v + h, u:u + w])
            save_png(name, row, clut[DATE_PAL * 16:DATE_PAL * 16 + 16])
    return pieces, (sx, sy, sw, sh)

def j(t): return 0x08000000 | (t >> 2 & 0x3ffffff)

class Asm2(Asm):
    def multu(self, rs, rt): self.r(0, rs, rt, 'zero', 0, 0x19)
    def mfhi(self, rd): self.r(0, 'zero', 'zero', rd, 0, 0x10)
    def slti(self, rt, rs, imm): self.i(0x0a, rs, rt, imm)
    def j(self, t): self.emit(j(t))

# Layout 1 digit switch (block start = 'addiu v0,zero,1' after the sprite
# pointers are set up; COMMON = 'addu v0,v0,s3'; SKIP = 'addiu s1,s1,1').
L1 = {'MGAME.BIN': 0x800b418c, 'MGFIN.BIN': 0x800b418c - 0xcd74}
L1_COMMON, L1_SKIP, L1_END = 0xdc, 0x188, 0xdc     # offsets from block start
L1_HEAD = [0x24020001, 0x12220019, 0x2a220002, 0x10400005]
# Layout 2 (MGAME calendar): jump table entry of case 3, common tail.
L2_TABLE3, L2_CASE3, L2_COMMON = 0x800c16b4, 0x800b8040, 0x800b80b0

def layout1_code(org, with_l2):
    a = Asm2(org)
    a.labels['COMMON'] = org + L1_COMMON; a.labels['SKIP'] = org + L1_SKIP
    a.lui('v0', 0x801f)
    a.srl('v1', 's1', 1)
    a.addu('v0', 'v0', 'v1')
    a.lbu('a0', 0x1851, 'v0')           # month (k 0,1) or day (k 2,3)
    a.addiu('a1', 'zero', 0x1f4)
    a.multu('a0', 's5')                 # s5 = 0xcccccccd
    a.mfhi('t0')
    a.srl('v1', 't0', 3)                # tens
    a.andi('v0', 's1', 1)
    a.branch('bne', 'v0', 'zero', 'ONES'); a.nop()
    a.branch('beq', 'v1', 'zero', 'SKIP'); a.nop()
    a.j(org + L1_COMMON); a.sll('v0', 'v1', 2)
    a.label('ONES')
    a.sll('v0', 'v1', 2); a.addu('v0', 'v0', 'v1'); a.sll('v0', 'v0', 1)
    a.subu('a0', 'a0', 'v0')
    a.andi('a0', 'a0', 0xff)
    a.slti('v0', 's1', 3)
    a.branch('bne', 'v0', 'zero', 'COMMON'); a.sll('v0', 'a0', 2)
    a.branch('bne', 'v1', 'zero', 'COMMON'); a.nop()
    a.addiu('s4', 's4', -0x3c)          # day < 10: use the tens sprite
    a.j(org + L1_COMMON); a.addiu('s2', 's2', -0x3c)
    l2 = None
    if with_l2:
        l2 = a.pc(); a.label('L2')
        a.lui('a0', 0x801f)
        a.lbu('a0', 0x1852, 'a0')
        a.addiu('a1', 'zero', 0x1f4)
        a.multu('a0', 's6')
        a.mfhi('a3')
        a.srl('v1', 'a3', 3)
        a.sll('v0', 'v1', 2); a.addu('v0', 'v0', 'v1'); a.sll('v0', 'v0', 1)
        a.subu('a0', 'a0', 'v0')
        a.branch('bne', 'v1', 'zero', 'L2B'); a.sll('v0', 'a0', 2)
        a.addiu('s2', 's2', -0x38)      # day < 10: use the tens sprite
        a.addiu('s1', 's1', -0x38)
        a.label('L2B')
        a.addu('v0', 'v0', 's5')
        a.lw('s0', 0, 'v0')
        a.lui('v0', 0x801f)
        a.lw('v0', 0x6370, 'v0')
        a.j(L2_COMMON); a.addiu('s0', 's0', 2)
    code = a.finish()
    assert len(code) <= L1_COMMON, len(code)
    return code + bytes(L1_COMMON - len(code)), l2

# Records (file offsets).
HDR = {'MGAME.BIN': 0x109c, 'MGFIN.BIN': 0x908}      # 月 日 ( ), 20 bytes
WDR = {'MGAME.BIN': 0x14cc, 'MGFIN.BIN': 0xd38}      # 7 x 20 bytes, Sun..Sat
DGX = {'MGAME.BIN': 0x1574, 'MGFIN.BIN': 0xde0}      # 4 x (x, y)
L2_HDR, L2_WDR, L2_DGX = 0x1b1c, 0x1db4, 0x1e24      # 16-byte, 12-byte, 5 x (x, y)

def patch_date(name, mg, pieces, slash):
    org = L1[name]; o = org - B
    head = list(struct.unpack_from('<4I', mg, o))
    if head != L1_HEAD:
        sys.exit(f'{name}: date code at {org:#x} is not the original; rerun hook.py and build_menus.py')
    assert struct.unpack_from('<I', mg, o + L1_COMMON)[0] == 0x00531021
    assert struct.unpack_from('<I', mg, o + L1_SKIP)[0] == 0x26310001
    code, l2 = layout1_code(org, name == 'MGAME.BIN')
    mg[o:o + len(code)] = code
    su, sv, sw, sh = slash
    h = HDR[name]
    def rec20(off, u, v, w, hh, x):
        tp, ty, cx, cy = struct.unpack_from('<4h', mg, off)
        assert (tp, ty, cx, cy) == (512, 256, 176, 500), (name, hex(off))
        y = struct.unpack_from('<h', mg, off + 18)[0]
        struct.pack_into('<6h', mg, off + 8, u, v, w, hh, x, y)
    rec20(h, su, sv, sw, sh, X1['slash'])
    for k in (1, 2, 3): rec20(h + 20 * k, BLANK[0], BLANK[1], 1, 1, 0)
    for d in range(7):
        u, v, w, hh = pieces[f'wd{d}']
        rec20(WDR[name] + 20 * d, u, v, w, hh, X1['wd'])
    struct.pack_into('<8h', mg, DGX[name], X1['m10'], 37, X1['m1'], 37, X1['d10'], 37, X1['d1'], 37)
    if name != 'MGAME.BIN': return
    # layout 2: calendar screen header, everything at x + 96
    assert struct.unpack_from('<I', mg, L2_TABLE3 - B)[0] == L2_CASE3
    struct.pack_into('<I', mg, L2_TABLE3 - B, l2)
    def rec16(off, u, v, w, hh, x):
        tp, cl = struct.unpack_from('<2h', mg, off)
        assert (tp, cl) == (8, 11), hex(off)
        struct.pack_into('<6h', mg, off + 4, u, v, w, hh, x, 37)
    rec16(L2_HDR, su, sv, sw, sh, X1['slash'] + 96)
    for k in (1, 2, 3): rec16(L2_HDR + 16 * k, BLANK[0], BLANK[1], 1, 1, 0)
    for d in range(7):
        u, v, w, hh = pieces[f'wd{d}']
        off = L2_WDR + 12 * d
        assert struct.unpack_from('<h', mg, off)[0] == 11, hex(off)
        struct.pack_into('<4h', mg, off + 2, u, v, w, hh)
    struct.pack_into('<10h', mg, L2_DGX, *[v for k in ('m10', 'm1', 'd10', 'd1', 'wd')
                                           for v in (X2_WD if k == 'wd' else X1[k] + 96, 37)])

# ---------------------------------------------------------- school names
SCHOOLS = [('cont_taiyo', 0), ('cont_gorin', 24), ('cont_gedo', 48),
           ('cont_pacific', 72), ('cont_justice', 96)]

def school_cell(text, cell, pal, ref):
    """80x24 label in the build_menus.py style (fill, mid edge, 1 px
    shadow) using the palette indices the original kanji used."""
    from build_menus import text_mask, lum
    used = [i for i, n in collections.Counter(ref.ravel().tolist()).items() if i and n > 2]
    ramp = sorted(used, key=lambda i: lum(pal[i]))
    a = np.zeros((24, 80))
    a[3:21] = text_mask(text, 80, 18)    # same size as the hub labels
    sh = np.zeros_like(a); sh[1:, 1:] = a[:-1, :-1]
    out = np.zeros((24, 80), np.uint8)
    out[sh > 0.4] = ramp[0]
    out[a > 0.25] = ramp[len(ramp) * 2 // 3]
    out[a > 0.55] = ramp[-1]
    return out

def build_schools_cock(tl, px, clut, png):
    """MA_COCK school cells (records at MGAME 0x3554 show them 80x24)."""
    pal = clut[5 * 16:5 * 16 + 16]
    for v, h in ((0, 18), (24, 18), (48, 18), (72, 18)):
        px[v + h:v + 24, 0:80] = 0       # kanji rows below the hub labels
    ref = px[96:120, 0:80].copy()
    img = school_cell(tl['cont_justice'], 'justice', pal, ref)
    px[96:120, 0:80] = img
    if png: save_png('cock_justice', img, pal)

PLAQUE_TIM, PLAQUE_PIX = 0x3800, 0x3a20         # 8bpp 280x240 photo
PLAQUE = (131, 208, 226, 218)                    # x0, y0, x1, y1 (inclusive)
PLAQUE_FONT = '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf'

def paint_plaque(d, text, png):
    """太陽学園 plate on the school photo: fill the plate with its own dark
    colour and draw the EN in the colour of the old engraving."""
    import build_banners as bb
    assert struct.unpack_from('<2I', d, PLAQUE_TIM) == (0x10, 9)
    clut = struct.unpack_from('<256H', d, PLAQUE_TIM + 20)
    pal = np.array([bb.rgb15(v) for v in clut])
    idx = np.frombuffer(bytes(d[PLAQUE_PIX:PLAQUE_PIX + 280 * 240]), np.uint8).reshape(240, 280).copy()
    img = pal[idx]
    x0, y0, x1, y1 = PLAQUE
    reg = img[y0:y1 + 1, x0:x1 + 1]
    L = reg.mean(-1)
    dark = np.median(reg[L < 45], axis=0)
    light = np.clip(reg[L >= np.percentile(L, 97)].max(0) * 1.1, 0, 255)
    plate = np.zeros(L.shape, bool)               # plate and engraving
    plate[:-2] = True
    plate[-2:] = L[-2:] < 45
    reg[plate] = dark
    ss = 4; H, W = reg.shape[:2]
    size = 8
    while True:
        f = ImageFont.truetype(PLAQUE_FONT, size * ss)
        b = f.getbbox(text)
        if b[3] - b[1] > 7 * ss or b[2] - b[0] > (W - 6) * ss: size -= 1; break
        size += 1
    f = ImageFont.truetype(PLAQUE_FONT, size * ss); b = f.getbbox(text)
    im = Image.new('L', (W * ss, H * ss), 0)
    ImageDraw.Draw(im).text(((W * ss - (b[2] - b[0])) // 2 - b[0],
                             (9 * ss - (b[3] - b[1])) // 2 - b[1] + ss // 2), text, font=f, fill=255)
    a = np.asarray(im.resize((W, H), Image.LANCZOS), float)[..., None] / 255
    reg[:] = reg * (1 - a) + light * a
    new = bb.quantize(np.clip(img, 0, 255), pal, idx)
    d[PLAQUE_PIX:PLAQUE_PIX + 280 * 240] = new.astype(np.uint8).tobytes()
    if png:
        os.makedirs(OUT, exist_ok=True)
        Image.fromarray(pal[new][196:228, 120:236].astype(np.uint8)).resize((696, 192), Image.NEAREST).save(
            f'{OUT}/plaque.png')

def build_ma_tit(tl, png):
    d = bytearray(open('private/build/MA_TIT.BIZ.lz', 'rb').read())
    paint_plaque(d, tl.get('plaque', 'TAIYO HIGH'), png)
    clut = struct.unpack_from('<4096H', d, 20)
    pal = clut[5 * 16:5 * 16 + 16]
    px = unpack4(d, 0x2020, 80, 120)
    for k, v in SCHOOLS:
        img = school_cell(tl[k], k, pal, px[v:v + 24].copy())
        px[v:v + 24] = img
    d[0x2020:0x2020 + 80 * 120 // 2] = pack4(px)
    if png: save_png('tit_names', px, pal, 4)
    c = lz.compress(bytes(d))
    slot = os.path.getsize('private/disc/DAT2/MA_TIT.BIZ'); slot += -slot % 2048
    if len(c) > slot: print(f'MA_TIT: {len(c)} > slot {slot}'); return 1
    write('private/build/MA_TIT.BIZ.lz', d)
    write('private/build/MA_TIT.BIZ', c)
    print(f'MA_TIT: {len(c)} bytes (slot {slot})')
    return 0

# ---------------------------------------------------------- club banner
MZP_TIM = 0x46800
CLUB_RECT = (76, 10, 436, 50)           # JP lettering, in the 512x52 canvas
CLUB_FILL = ((248, 200, 40), (248, 136, 24))    # top -> bottom
CLUB_LINE = (232, 44, 0)
CLUB_FONT = '/usr/share/fonts/TTF/ariblk.ttf'

def row_inpaint(img, mk):
    """Fill masked runs of each row by linear interpolation between the
    unmasked pixels left and right of them (the background is a set of
    horizontal streaks), then smooth the filled pixels a little."""
    out = img.copy()
    for y in range(img.shape[0]):
        m = mk[y]
        if not m.any(): continue
        xs = np.arange(img.shape[1]); good = ~m
        for c in range(3):
            out[y, m, c] = np.interp(xs[m], xs[good], img[y, good, c])
    sm = ndimage.uniform_filter(out, size=(3, 3, 1))
    out[mk] = sm[mk]
    return out

def club_text(img, text, cy, cap, maxw, outline=2, ss=4):
    size = 10
    while ImageFont.truetype(CLUB_FONT, size * ss).getbbox('H')[3] - \
            ImageFont.truetype(CLUB_FONT, size * ss).getbbox('H')[1] < cap * ss: size += 1
    f = ImageFont.truetype(CLUB_FONT, size * ss)
    bb_ = f.getbbox(text); asc = f.getmetrics()[0]
    W = bb_[2] - bb_[0] + 8 * ss; H = img.shape[0] * ss
    im = Image.new('L', (W, H), 0)
    base = int((cy + cap / 2) * ss)
    ImageDraw.Draw(im).text((4 * ss - bb_[0], base - asc), text, font=f, fill=255)
    w = W // ss
    sx = min(1.0, (maxw - 2 * outline) / (w - 8))
    nw = int(round(w * sx))
    a = np.asarray(im.resize((nw, img.shape[0]), Image.LANCZOS), float) / 255
    st = ndimage.generate_binary_structure(2, 2)
    o = a.copy()
    for _ in range(outline): o = ndimage.grey_dilation(o, footprint=st)
    x = int(round(256 - nw / 2))
    reg = img[:, x:x + nw]
    ys = np.where(a.max(1) > 0.5)[0]
    t = np.clip((np.arange(img.shape[0]) - ys[0]) / max(1, ys[-1] - ys[0]), 0, 1)[:, None, None]
    fill = np.array(CLUB_FILL[0]) * (1 - t) + np.array(CLUB_FILL[1]) * t
    line = np.array(CLUB_LINE, float)
    reg[:] = reg * (1 - o[..., None]) + line * o[..., None]
    reg[:] = reg * (1 - a[..., None]) + fill * a[..., None]
    return sx

def paint_club(d, tl, png, tag):
    """Repaint the banner in one decompressed MZPANEL entry (in place)."""
    import build_banners as bb
    assert struct.unpack_from('<2I', d, MZP_TIM) == (0x10, 9)
    clut = struct.unpack_from('<256H', d, MZP_TIM + 20)
    pix = MZP_TIM + 8 + 524 + 12
    idx = np.frombuffer(bytes(d[pix:pix + 256 * 104]), np.uint8).reshape(104, 256)
    can = np.concatenate([idx[:52], idx[52:]], 1).copy()        # 52 x 512
    pal = np.array([bb.rgb15(v) for v in clut])
    img = pal[can]
    x0, y0, x1, y1 = CLUB_RECT
    mk = np.zeros(can.shape, bool)
    sub = img[y0:y1, x0:x1]
    mk[y0:y1, x0:x1] = (sub[..., 0] > 110) & (sub[..., 2] < 160)
    mk = ndimage.binary_dilation(ndimage.binary_closing(mk), iterations=1)
    img = row_inpaint(img, mk)
    text = tl['club_banner'].replace('|', ' ')
    sx = club_text(img, text, 29.5, 22, bb.VIS[1] - bb.VIS[0] - 40)
    if sx < 0.8: print(f'club banner: squeezed to {sx:.2f}')
    new = bb.quantize(np.clip(img, 0, 255), pal, can)
    idx2 = np.concatenate([new[:, :256], new[:, 256:]], 0)
    d[pix:pix + 256 * 104] = idx2.astype(np.uint8).tobytes()
    if png:
        os.makedirs(OUT, exist_ok=True)
        Image.fromarray(pal[new].astype(np.uint8)).resize((1024, 104), Image.NEAREST).save(
            f'{OUT}/club{tag}.png')

def build_mzpanel(tl, png, caltex):
    """Club banner, calendar labels and the logo tile (0x41000, drawn by
    build_logotile.patch) in MZPANEL entries 0-2 (entry 0
    from build_menus.py's output, 1 and 2 from the disc). All three carry
    the banner and the calendar TIMs; training days load entry 1 or 2."""
    src0 = 'private/build/MZPANEL.BIZ.0.lz'
    if not os.path.exists(src0): sys.exit('run build_menus.py first (no ' + src0 + ')')
    arc = bytearray(open('private/build/MZPANEL.BIZ', 'rb').read())
    toc = [struct.unpack_from('<II', arc, 8 * i) for i in range(3)]
    ends = [toc[1][0], toc[2][0], len(arc)]
    problems = 0
    for e in range(3):
        src = src0 if e == 0 else f'private/unpacked/MZPANEL.BIZ/{e:03d}.lz'
        d = bytearray(open(src, 'rb').read())
        paint_club(d, tl, png and e == 0, '')
        assert struct.unpack_from('<4H', d, CAL_PIX - 8) == (896, 0, 128, 256)
        d[CAL_PIX:CAL_PIX + 65536] = pack4(caltex)
        import build_logotile                   # 私立ジャスティス学園 tile (0x41000)
        build_logotile.patch(d, 0x41000)
        c = lz.compress(bytes(d))
        o0, slot = toc[e][0], ends[e] - toc[e][0]
        if len(c) > slot: print(f'MZPANEL entry {e}: {len(c)} bytes > slot {slot}'); problems += 1; continue
        arc[o0:o0 + slot] = c + bytes(slot - len(c))
        struct.pack_into('<I', arc, 8 * e + 4, len(c))
        if e == 0: write(src0, d)
        print(f'MZPANEL entry {e}: {len(c)} bytes (slot {slot})')
    write('private/build/MZPANEL.BIZ', arc)
    return problems

# ------------------------------------------------------ calendar labels
# private/re/calendar/notes.org. Label TIM (4bpp 512x256 at VRAM 896,0,
# tpages 14/15) is redrawn from scratch: 15 two-line 116x40 boxes and a
# 122 px cursor. Event table order (MGAME 0x1f78); EN from DIARYGFX
# cal00-cal14 when present, else these defaults (BANNERS.txt wording).
CAL_PIX = 0x1ec20
CAL_EV, CAL_REC, CAL_CUR, CAL_HDR, CAL_HREC = 0x1f78, 0x1c7c, 0x1b74, 0x1e64, 0x1da8
CAL_FONT = '/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf'
CAL_W, CAL_H = 116, 40
CAL_DEFAULT = [('Entrance|Ceremony', 140), ('1st Term|Midterm Exam', 140),
               ('Taiyo-Pacific|Match', 140), ('1st Term|Final Exam', 140),
               ('Seaside|Camp', 140), ('School|Festival', 140),
               ('2nd Term|Midterm Exam', 140), ('Gorin|Festival', 140),
               ('2nd Term|Final Exam', 140), ('Christmas', 192),
               ('New Year', 140), ("Valentine's|Day", 140),
               ('Year-End|Exam', 136), ('Promotion|Test', 184),
               ('Closing|Ceremony', 232)]

def cal_alpha(lines, w, h, size=15, ss=4):
    f = ImageFont.truetype(CAL_FONT, size * ss)
    im = Image.new('L', (w * ss, h * ss), 0); dr = ImageDraw.Draw(im)
    pitch = h * ss // 2 if len(lines) == 2 else h * ss
    over = []
    for i, t in enumerate(lines):
        b = f.getbbox(t); tw, th = b[2] - b[0], b[3] - b[1]
        if tw > (w - 4) * ss: over.append(f'{t} {tw / ss:.0f}px > {w - 4}')
        dr.text(((w * ss - tw) // 2 - b[0], i * pitch + (pitch - th) // 2 - b[1]), t, font=f, fill=255)
    return np.asarray(im.resize((w, h), Image.LANCZOS)) / 255.0, over

def patch_calendar(tl, mg, png):
    """Returns the new label texture; edits the MGAME tables in place."""
    from build_menus import lum
    d = open('private/unpacked/MZPANEL.BIZ/000.lz', 'rb').read()
    cl = struct.unpack_from('<512H', d, 0x1e814)
    cluts = [cl[16 * k:16 * k + 16] for k in range(32)]
    old = unpack4(d, CAL_PIX, 512, 256)
    tex = np.zeros_like(old)
    slots = [(14, (k % 2) * 120, (k // 2) * 40) for k in range(12)] + \
            [(15, 0, 0), (15, 0, 40), (15, 0, 80)]
    problems = 0
    for i, (dflt, y) in enumerate(CAL_DEFAULT):
        text = tl.get(f'cal{i:02d}', dflt)
        e = list(struct.unpack_from('<HhhhhBB', mg, CAL_EV + 12 * i))
        ro = CAL_REC + 12 * e[0]
        tp, c = struct.unpack_from('<hh', mg, ro)
        pal = cluts[c]
        nz = range(1, 16)
        white = max(nz, key=lambda k: lum(pal[k])); dark = min(nz, key=lambda k: lum(pal[k]))
        mid = sorted(nz, key=lambda k: lum(pal[k]))[7]
        a, over = cal_alpha(text.split('|'), CAL_W, CAL_H)
        for o in over: print(f'cal{i:02d}: {o}'); problems += 1
        img = np.zeros((CAL_H, CAL_W), np.uint8)
        img[ndimage.binary_dilation(a > 0.3, np.ones((3, 3), bool))] = dark
        img[a > 0.3] = mid
        img[a > 0.55] = white
        tpn, u, v = slots[i]
        tx = u + (256 if tpn == 15 else 0)
        tex[v:v + CAL_H, tx:tx + CAL_W] = img
        struct.pack_into('<6h', mg, ro, tpn, c, u, v, CAL_W, CAL_H)
        month = e[5] if e[5] < 128 else e[5] - 256              # 0 = April
        e[1] = 128 * (month % 12) + 6
        e[2], e[3], e[4] = y, -5, -15
        struct.pack_into('<HhhhhBB', mg, CAL_EV + 12 * i, *e)
        if png: save_png(f'cal{i:02d}', img, pal, 2)
    # cursor: stretch the original 40x24 pieces (36 px visible) to W + 6
    BW = CAL_W + 6
    for piece, v0 in ((0, 152), (1, 176)):
        ro = CAL_CUR + 12 * piece
        assert struct.unpack_from('<6h', mg, ro) == (14, 25, 0, v0, 40, 24), hex(ro)
        src = old[v0:v0 + 24, 0:36]
        dst = np.zeros((24, BW), np.uint8)
        dst[:, :3] = src[:, :3]; dst[:, BW - 3:] = src[:, 33:36]
        dst[:, 3:BW - 3] = src[:, 10:11]
        tex[160 + 24 * piece:184 + 24 * piece, 384:384 + BW] = dst
        struct.pack_into('<6h', mg, ro, 15, 25, 128, 160 + 24 * piece, BW, 24)
    # month headers: "N" + 月 -> the number alone, centred over the tile
    assert struct.unpack_from('<6h', mg, CAL_HREC) == (8, 11, 48, 120, 24, 24)
    struct.pack_into('<6h', mg, CAL_HREC, 8, 11, BLANK[0], BLANK[1], 1, 1)
    hdr = [list(struct.unpack_from('<4h', mg, CAL_HDR + 8 * k)) for k in range(27)]
    groups, cur = [], []
    for h in hdr:
        cur.append(h)
        if h[0] == 10: groups.append(cur); cur = []
    assert len(groups) == 12 and not cur
    for t, g in enumerate(groups):
        digs = g[:-1]
        x = 128 * t + (52 if len(digs) == 1 else 45)
        for k, h in enumerate(digs): h[1] = x + 14 * k
    for k, h in enumerate(hdr): struct.pack_into('<4h', mg, CAL_HDR + 8 * k, *h)
    if png:
        os.makedirs(OUT, exist_ok=True)
        Image.fromarray((tex * 16).astype(np.uint8)).save(f'{OUT}/cal_tex.png')
    return tex, problems

# ------------------------------------------------------------------ main
def build(png=False):
    tl = load_tl()
    problems = 0
    m = bytearray(open('private/build/MA_COCK.BIZ.lz', 'rb').read())
    clut = struct.unpack_from('<4096H', m, 20)
    px = unpack4(m, 0x2020, 1024, 256)
    pieces, slash = build_date_gfx(tl, px, clut, png)
    build_schools_cock(tl, px, clut, png)
    for name in ('MGAME.BIN', 'MGFIN.BIN'):
        mg = bytearray(open('private/build/' + name, 'rb').read())
        patch_date(name, mg, pieces, slash)
        if name == 'MGAME.BIN':
            caltex, p = patch_calendar(tl, mg, png); problems += p
        write('private/build/' + name, mg)
    m[0x2020:0x2020 + 131072] = pack4(px)
    c = lz.compress(bytes(m))
    slot = os.path.getsize('private/disc/DAT2/MA_COCK.BIZ'); slot += -slot % 2048
    if len(c) > slot: print(f'MA_COCK: {len(c)} > slot {slot}'); problems += 1
    else:
        write('private/build/MA_COCK.BIZ.lz', m)
        write('private/build/MA_COCK.BIZ', c)
        print(f'MA_COCK: {len(c)} bytes (slot {slot})')
    problems += build_mzpanel(tl, png, caltex)
    problems += build_ma_tit(tl, png)
    print(f'diarygfx: problems {problems}')
    return problems

if __name__ == '__main__':
    DRY = '--dry' in sys.argv
    sys.exit(1 if build('--png' in sys.argv) else 0)
