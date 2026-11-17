# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Redraw the diary-mode label graphics (DAT2/MA_COCK.BIZ, TIM 1) in English.

Reads tl/MENUS.txt (ID: EN) and data/menu_labels.tsv (record
offsets and texel rectangles). Writes private/build/MA_COCK.BIZ (+ .lz) and
updates the sprite records in private/build/MGAME.BIN and their copies in
private/build/MGFIN.BIN (year-end overlay), so run it after hook.py. The
records it owns are reset from the disc MGAME first, so a rerun gives the
same result.
--png writes a 3x preview of every redrawn label to private/re/menus/en/.

Texture: 1024x256 4bpp at file offset 0x2020, 512 bytes per row. Each
label keeps its palette; the text uses the brightest colour the original
used, edges a mid tone and a 1 px drop shadow the darkest one.
"""
import sys, os, re, csv, struct, collections
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import lz

SRC = 'private/unpacked/MA_COCK.BIZ.lz'
PIX = 0x2020
FONT = '/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf'
# Hub header labels (school, club, training) and the stat row: Roboto
# Condensed Bold 13 (OFL 1.1), hinted 1x.
HUB_FONT, HUB_SIZE = '/usr/share/fonts/TTF/RobotoCondensed-Bold.ttf', 13

def load_tl(path='tl/MENUS.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        k, v = line.split(':', 1)
        out[k.strip()] = v.strip()
    return out

def lum(v): return (v & 31) * 2 + (v >> 5 & 31) * 3 + (v >> 10 & 31)

def text_mask(text, w, h, ss=4):
    """Alpha mask (h, w) in 0..1 with the text centred, shrunk to fit."""
    size = h * 0.78 if h <= 24 else 22
    while True:
        f = ImageFont.truetype(FONT, int(size * ss))
        x0, y0, x1, y1 = f.getbbox(text)
        tw, th = x1 - x0, y1 - y0
        im = Image.new('L', (tw + 2 * ss, th + 2 * ss), 0)
        ImageDraw.Draw(im).text((ss - x0, ss - y0), text, font=f, fill=255)
        # target box leaves 1 px for the shadow
        sx = min(1.0, (w - 2) * ss / im.width)
        if sx >= 0.62 or size < 9: break
        size -= 1
    nw = max(1, int(im.width * sx / ss)); nh = max(1, int(im.height / ss))
    nh = min(nh, h - 1)
    small = im.resize((nw, nh), Image.LANCZOS)
    a = np.zeros((h, w))
    ox, oy = 0, (h - 1 - nh) // 2       # left-aligned like the originals
    a[oy:oy + nh, ox:ox + nw] = np.asarray(small) / 255.0
    return a

def draw(px, clut, row, text):
    x, y, w, h, pal = (int(row[k]) for k in ('texel_x', 'texel_y', 'w', 'h', 'pal'))
    pc = clut[pal * 16:pal * 16 + 16]
    used = [i for i, n in collections.Counter(px[y:y+h, x:x+w].ravel().tolist()).items()
            if i and n > 2]
    ramp = sorted(used, key=lambda i: lum(pc[i]))
    fill, shadow, mid = ramp[-1], ramp[0], ramp[len(ramp) * 2 // 3]
    a = text_mask(text, w, h)
    sh = np.zeros_like(a); sh[1:, 1:] = a[:-1, :-1]
    out = np.zeros((h, w), np.uint8)
    out[sh > 0.4] = shadow
    out[a > 0.25] = mid
    out[a > 0.55] = fill
    px[y:y+h, x:x+w] = out
    return out, pc

# Stat names in the hub's top-right box. Records at MGAME file 0xaa4, 5 x 16
# bytes: s16 tpage/64, clut/16 (y 500), u, v, w, h, screen x, y. The sprite
# is drawn 1:1 at w x h. All five labels use one font size and one baseline
# row, and every record gets the same h and screen y. The labels have one
# label's ink is centred on its bar (bar k spans x 268 + 47k .. +31) and
# moved only to keep STAT_GAP px to its neighbours and the row inside the
# box (x 262-489).
STAT_REC, STAT_Y, STAT_GAP = 0xaa4, 90, 6
STAT_BAR_C = [283.5 + 47 * k for k in range(5)]
STAT_X = (262, 490)
# Texel slots (tpage 512): the old rectangles, and only for labels wider
# than 44 px the free strip u 96-183, v 240-255 (below the date digits).
STAT_SLOTS = [(0, 120 + 24 * k, 44) for k in range(5)]
STAT_WIDE = (96, 240, 88)

def coverage(text, top, h, track=0.0, path=HUB_FONT, size=HUB_SIZE):
    """Hinted 1x coverage (h, w) of text, cropped to the ink plus the 1 px
    shadow column. The baseline is at row -top. track (px per letter, may
    be fractional) moves each glyph origin, rounded to whole px."""
    f = ImageFont.truetype(path, size)
    im = Image.new('L', (12 * len(text) + 20, h), 0)
    if track == 0:
        ImageDraw.Draw(im).text((2, -top), text, font=f, fill=255, anchor='ls')
    else:
        for i, ch in enumerate(text):
            if ch == ' ': continue
            x = f.getlength(text[:i + 1]) - f.getlength(ch) + i * track
            g = Image.new('L', im.size, 0)          # own layer, max-merged
            ImageDraw.Draw(g).text((2 + round(x), -top), ch, font=f, fill=255, anchor='ls')
            im = Image.fromarray(np.maximum(np.asarray(im), np.asarray(g)))
    a = np.asarray(im) / 255.0
    cols = np.where((a > 0.25).any(0))[0]
    return a[:, cols[0]:cols[-1] + 2]

def vmetrics(texts, path=HUB_FONT, size=HUB_SIZE):
    """Common top row (relative to the baseline) and piece height (+ the
    shadow row) for a set of labels."""
    f = ImageFont.truetype(path, size)
    boxes = [f.getbbox(t, anchor='ls') for t in texts]
    top = min(b[1] for b in boxes)
    return top, max(b[3] for b in boxes) - top + 1

def shade(a, fill, mid, shadow):
    """Palette indices of a coverage piece with a 1 px drop shadow."""
    sh = np.zeros_like(a); sh[1:, 1:] = a[:-1, :-1]
    out = np.zeros(a.shape, np.uint8)
    out[sh > 0.4] = shadow
    out[a > 0.25] = mid
    out[a > 0.55] = fill
    return out

def label_ramp(px, clut, r):
    """(fill, mid, shadow) indices the original label used, and its palette."""
    pal = int(r['pal']); pc = clut[pal * 16:pal * 16 + 16]
    x, y, w, h = (int(r[k]) for k in ('texel_x', 'texel_y', 'w', 'h'))
    used = [i for i, n in collections.Counter(px[y:y+h, x:x+w].ravel().tolist()).items()
            if i and n > 2]
    ramp = sorted(used, key=lambda i: lum(pc[i]))
    return (ramp[-1], ramp[len(ramp) * 2 // 3], ramp[0]), pc

def stat_layout(ws):
    """Screen x of each label. Its ink (piece minus the shadow column) is
    centred on its bar. Labels move only where they would come closer than
    STAT_GAP px or leave STAT_X: least-squares shift under those limits
    (isotonic regression of x minus the packed offset, clamped)."""
    off = [sum(ws[:k]) + k * STAT_GAP for k in range(5)]
    ideal = [c - (w - 2) / 2 - o for c, w, o in zip(STAT_BAR_C, ws, off)]
    blocks = []                                   # pool adjacent violators
    for v in ideal:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, n2 = blocks.pop(); v1, n1 = blocks.pop()
            blocks.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    y = [v for v, n in blocks for _ in range(n)]
    lo, hi = STAT_X[0], STAT_X[1] - ws[4] - off[4]
    if lo > hi: return None
    return [int(round(min(max(v, lo), hi))) + o for v, o in zip(y, off)]

def build_stats(tl, px, clut, mg, rows, png):
    ids = [f'stat{k}' for k in range(5)]
    texts = [tl.get(i, '=') for i in ids]
    if '=' in texts: return 0
    (fill, mid, shadow), pc = label_ramp(px, clut, rows['stat0'])
    top, H = vmetrics(texts)
    imgs = [shade(coverage(t, top, H), fill, mid, shadow) for t in texts]
    ws = [im.shape[1] for im in imgs]
    xs = stat_layout(ws)
    problems = 0
    if xs is None: print('stat labels do not fit the box'); return 1
    for k, (u, v, w) in enumerate(STAT_SLOTS):         # free the old rectangles
        px[v:v + int(rows[ids[k]]['h']), u:u + w] = 0
    wide = STAT_WIDE[0]
    for k, img in enumerate(imgs):
        w = img.shape[1]
        u, v, sw = STAT_SLOTS[k]
        if w > sw or H > 24:
            u, v = wide, STAT_WIDE[1]
            if H > 16 or wide + w > STAT_WIDE[0] + STAT_WIDE[2]:
                print(ids[k], 'no free slot'); problems += 1; continue
            assert not px[v:v + H, u:u + w].any(), 'stat wide slot not free'
            wide += w
        px[v:v + H, u:u + w] = img
        o = STAT_REC + 16 * k
        struct.pack_into('<6h', mg, o + 4, u, v, w, H, xs[k], STAT_Y)
        if png: preview(ids[k], img, pc)
    print('stat x', xs, 'ink centre - bar centre',
          [x + (w - 2) / 2 - c for x, w, c in zip(xs, ws, STAT_BAR_C)])
    return problems

# Hub top-left box: school, club and training labels (20-byte records, s16
# tpage_x, tpage_y, clut_x, clut_y, u, v, w, h, x, y) and the school emblem.
# All 18 labels share one font, one 14-row piece and one baseline (row 10).
# Row 2 (club) sits on the nickname baseline 82, row 3 (school, training)
# on 100. Every sprite draws 1:1 (record w, h = piece size). The pieces go
# into the old texel cells (v from labels.tsv), cleared first: school u 0
# (80 wide), club u 80, training u 152 (72 wide), 24 rows each. The training cells are also the
# training menu items (menu17-23, 72x24), so the piece sits 5 rows down
# to stay vertically centred there. Tracking (in -0.1 px steps, down to
# HUB_TRACK) is used only on labels wider than their cell.
HUB = {  # cat: (tl id fmt, tl id base, rows id fmt, first record, n, u, cell w, v in cell, x, baseline)
    'school': ('school%d', 0, 'school%d', 0x121c, 4, 0, 80, 0, 31, 100),
    'club': ('club%d', 0, 'club%d', 0x12dc, 7, 80, 72, 0, 131, 82),
    'train': ('menu%d', 17, 'train%d', 0x1384, 7, 152, 72, 5, 131, 100),
}
HUB_TRACK = -0.6
EMBLEM_REC, EMBLEM_Y = 0x127c, 84       # 4 x 20 bytes; y at +18 (x 105 kept)

def hub_ids():
    # tl ids drawn by build_hub()
    return {h[0] % (h[1] + k) for h in HUB.values() for k in range(h[4])}

def build_hub(tl, px, clut, mg, rows, png):
    jobs = []
    for cat, (tf, t0, rf, rec, n, u, cw, dv, x, base) in HUB.items():
        for k in range(n):
            tid = tf % (t0 + k)
            text = tl.get(tid, '=')
            if text == '=': print(tid, 'missing in MENUS.txt'); return 1
            r = rows[rf % k]
            assert int(r['texel_y']) % 24 == 0 and int(r['texel_x']) == u, (cat, k)
            jobs.append((cat, k, tid, r, text))
    top, H = vmetrics([j[4] for j in jobs])
    ramps = [label_ramp(px, clut, j[3]) for j in jobs]      # before clearing
    for cat, k, tid, r, text in jobs:
        u, cw = HUB[cat][5], HUB[cat][6]
        v = int(r['texel_y']); px[v:v + 24, u:u + cw] = 0
    problems = 0
    for (cat, k, tid, r, text), ((fill, mid, shadow), pc) in zip(jobs, ramps):
        tf, t0, rf, rec, n, u, cw, dv, x, base = HUB[cat]
        tr = 0.0
        a = coverage(text, top, H)
        while a.shape[1] > cw and tr > HUB_TRACK + 1e-6:
            tr = round(tr - 0.1, 1)
            a = coverage(text, top, H, tr)
        w = a.shape[1]
        if w > cw: print(tid, f'{text!r} is {w} px, cell {cw}'); problems += 1; continue
        if tr: print(f'{tid} {text!r}: tracking {tr}, {w} px')
        img = shade(a, fill, mid, shadow)
        v = int(r['texel_y']) + dv
        px[v:v + H, u:u + w] = img
        o = rec + 20 * k
        assert struct.unpack_from('<2h', mg, o) == (512, 256), tid
        struct.pack_into('<6h', mg, o + 8, u, v, w, H, x, base + top)
        if png: preview(tid if cat != 'train' else f'train{k}', img, pc)
    for k in range(4):
        struct.pack_into('<h', mg, EMBLEM_REC + 20 * k + 18, EMBLEM_Y)
    return problems

# Location titles. Records at MGAME file 0x167c, 16 bytes each:
# s16 tpage_x/64, clut_x/16 (y 500), u, v, w, h, dx, dy. The title box is
# 96x40 with a 3 px bevel (usable dx 3-93). Lists of record indices per
# location id (4 bytes, ff-terminated, record 0 = empty box last) at 0x194c.
LOC_REC, LOC_LIST = 0x167c, 0x194c
# Free 80-88 px texel slots (tpage/64, u, v, w) for titles whose own
# rectangle is narrower than 80 px. tpage 704 v110-175 is unused (the club
# animation overwrites v176+); loc06 and loc08 are freed by the composite
# rewrite below.
LOC_SLOTS = [(11, 0, 112, 88), (11, 88, 112, 88), (11, 176, 112, 80),
             (11, 0, 144, 88), (11, 88, 144, 88), (11, 176, 144, 80), (9, 184, 136, 72)]
# Composite titles become one record: id -> record (1F/2F hallway, rooms).
LOC_COMPOSITE = {7: 35, 10: 35, 22: 35, 23: 35, 8: 34, 24: 34, 25: 34, 61: 11, 62: 10}

def title_img(lines, w, h, pc, used, size):
    from build_title import mask
    a = mask(lines, w, h, size, align='centre', pitch=h // len(lines))
    ramp = sorted(used, key=lambda i: lum(pc[i]))
    sh = np.zeros_like(a); sh[1:, 1:] = a[:-1, :-1]
    out = np.zeros((h, w), np.uint8)
    out[sh > 0.4] = ramp[0]
    out[a > 0.25] = ramp[len(ramp) * 2 // 3]
    out[a > 0.55] = ramp[-1]
    return out

def preview(name, img, pc):
    rgb = np.array([[(c & 31) * 8, (c >> 5 & 31) * 8, (c >> 10 & 31) * 8] for c in pc], np.uint8)
    Image.fromarray(rgb[img]).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(
        f'private/re/menus/en/{name}.png')

def build_loc(tl, px, clut, mg, png):
    problems = 0
    slots = list(LOC_SLOTS)
    for idn, rec in LOC_COMPOSITE.items():
        mg[LOC_LIST + 4 * idn:LOC_LIST + 4 * idn + 3] = bytes([rec, 0, 0xff])
    # composites first so they are sure to get a slot
    for k in [34, 35, 10, 11] + [k for k in range(1, 36) if k not in (34, 35, 10, 11)]:
        rid = f'loc{k:02d}'
        if rid not in tl or tl[rid] == '=': continue
        o = LOC_REC + 16 * k
        tp, cl, u, v, w, h, dx, dy = struct.unpack_from('<8h', mg, o)
        tx = (tp * 64 - 512) * 4 + u
        used = [i for i, n in collections.Counter(px[v:v+h, tx:tx+w].ravel().tolist()).items() if i and n > 2]
        if w < 80 and max(len(t) for t in tl[rid].split('|')) > 6:
            if not slots: print(rid, 'no free slot'); problems += 1; continue
            tp, u, v, w = slots.pop(0); h = 32
            tx = (tp * 64 - 512) * 4 + u
        pc = clut[cl * 16:cl * 16 + 16]
        lines = tl[rid].split('|')
        img = title_img(lines, w, h, pc, used, 20 if len(lines) == 1 else 14)
        px[v:v+h, tx:tx+w] = img
        dx = 3 + (90 - w) // 2
        struct.pack_into('<8h', mg, o, tp, cl, u, v, w, h, dx, 4)
        if png: preview(rid, img, pc)
    return problems

# School-map room labels: DAT2/MZPANEL.BIZ entry 0, TIM at 0x68800 (CLUT
# row at VRAM y 481, 4bpp 512x256 at VRAM 640,0, pixels at 0x68a20).
# Per-panel MGAME tables: label screen xy 0x29a8 (2 s16), uv 0x29bc (2 u8),
# wh 0x29c8 (2 s16), tpage/64 0x29dc (u8). Palette pal 1 (pal 10 when the
# panel is chosen; same indices). New labels go to tpage 704 v88-175.
MZP = 'private/unpacked/MZPANEL.BIZ/000.lz'
MZP_TIM, MZP_PIX = 0x68800, 0x68a20
MAP_MAXW = [112, 112, 48, 112, 56]

def build_map(tl, mg, png):
    if not any(k.startswith('map') for k in tl): return 0
    import lz
    d = bytearray(open(MZP, 'rb').read())
    clut = struct.unpack_from('<256H', d, MZP_TIM + 8 + 12)
    pc = clut[16:32]
    raw = np.frombuffer(bytes(d[MZP_PIX:MZP_PIX + 65536]), np.uint8).reshape(256, 256)
    px = np.empty((256, 512), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    used = [i for i, n in collections.Counter(px[56:80, 184:224].ravel().tolist()).items() if i and n > 2]
    gold = sorted([i for i in used if lum(pc[i]) > lum(pc[9]) * 0.5], key=lambda i: lum(pc[i]))
    dark = min(used, key=lambda i: lum(pc[i]))
    for k in range(5):
        rid = f'map{k}'
        if rid not in tl: continue
        u, v, w, h = (k % 2) * 112, 88 + (k // 2) * 24, MAP_MAXW[k], 24
        tx = 256 + u
        from build_title import mask
        a = mask([tl[rid]], w, h, 17)
        sh = np.zeros_like(a); sh[1:, 1:] = a[:-1, :-1]
        img = np.zeros((h, w), np.uint8)
        img[sh > 0.4] = dark; img[a > 0.25] = gold[len(gold) // 2]; img[a > 0.55] = gold[-1]
        px[v:v+h, tx:tx+w] = img
        mg[0x29bc + 2 * k] = u; mg[0x29bc + 2 * k + 1] = v
        struct.pack_into('<2h', mg, 0x29c8 + 4 * k, w, h)
        mg[0x29dc + k] = 11
        if png: preview(rid, img, pc)
    d[MZP_PIX:MZP_PIX + 65536] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()
    c = lz.compress(bytes(d))
    arc = bytearray(open('private/disc/DAT2/MZPANEL.BIZ', 'rb').read())
    o0, s0 = struct.unpack_from('<II', arc, 0); o1 = struct.unpack_from('<I', arc, 8)[0]
    if o0 + len(c) > o1: print(f'MZPANEL entry 0: {len(c)} bytes > slot {o1 - o0}'); return 1
    arc[o0:o1] = c + bytes(o1 - o0 - len(c))
    struct.pack_into('<I', arc, 4, len(c))
    open('private/build/MZPANEL.BIZ.0.lz', 'wb').write(d)
    open('private/build/MZPANEL.BIZ', 'wb').write(arc)
    print(f'MZPANEL entry 0: compressed {len(c)} bytes (slot {o1 - o0})')
    return 0

# MGFIN.BIN (year-end overlay, loaded by the exe after the staff roll, state
# 12) has its own copies of the hub tables. MGAME file offset - delta:
# stat records 0x60c, school/club/training records 0x794 (same block as
# the date header), location title records and lists 0x800. MGFIN has no
# copy of the menu labels, map tables or UP tables.
FIN_DELTA = {'stat': 0x60c, 'school': 0x794, 'club': 0x794, 'train': 0x794, 'loc': 0x800}
RSIZE = {'menu': 16, 'stat': 16, 'school': 20, 'club': 20, 'train': 20, 'loc': 16}
DISC_MGAME, DISC_MGFIN = 'private/disc/DAT1/MGAME.BIN', 'private/disc/DAT1/MGFIN.BIN'
MAP_TAB = (0x29a8, 0x29e1)              # map xy, uv, wh, tpage tables

def owned(rows):
    """(MGAME file offset, size, MGFIN delta or None) of every table this
    tool writes."""
    out = []
    for rid, r in rows.items():
        cat = re.match(r'[a-z]+', rid)[0]
        out.append((int(r['rec_file'], 16), RSIZE[cat], FIN_DELTA.get(cat)))
    out.append((EMBLEM_REC, 20 * 4, 0x794))
    out.append((LOC_REC, 16 * 36, 0x800))
    out.append((LOC_LIST, 4 * 96, 0x800))
    out.append((MAP_TAB[0], MAP_TAB[1] - MAP_TAB[0], None))
    return out

def reset_mgame(mg, regions):
    disc = open(DISC_MGAME, 'rb').read()
    for o, n, _ in regions: mg[o:o + n] = disc[o:o + n]

def sync_mgfin(mg, regions):
    """Copy the final MGAME tables into private/build/MGFIN.BIN."""
    a, f = open(DISC_MGAME, 'rb').read(), open(DISC_MGFIN, 'rb').read()
    fin = bytearray(open('private/build/MGFIN.BIN', 'rb').read())
    n_rec = 0
    for o, n, dl in regions:
        if dl is None: continue
        assert a[o:o + n] == f[o - dl:o - dl + n], f'MGFIN copy of MGAME {o:#x} moved'
        fin[o - dl:o - dl + n] = mg[o:o + n]
        n_rec += 1
    open('private/build/MGFIN.BIN', 'wb').write(fin)
    return n_rec

def build(png=False):
    tl = load_tl()
    rows = {r['id']: r for r in csv.DictReader(open('data/menu_labels.tsv'), delimiter='\t')}
    d = bytearray(open(SRC, 'rb').read())
    clut = struct.unpack_from('<4096H', d, 20)
    raw = np.frombuffer(bytes(d[PIX:PIX + 131072]), np.uint8).reshape(256, 512)
    px = np.empty((256, 1024), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    mg = bytearray(open('private/build/MGAME.BIN', 'rb').read())
    regions = owned(rows)
    reset_mgame(mg, regions)
    problems = 0
    if png: os.makedirs('private/re/menus/en', exist_ok=True)
    problems += build_hub(tl, px, clut, mg, rows, png)
    hub = hub_ids()
    for rid, text in tl.items():
        if rid.startswith(('loc', 'map', 'stat')) or rid in hub: continue
        if rid not in rows: print('unknown id', rid); problems += 1; continue
        if text == '=': continue
        r = rows[rid]
        x, w = int(r['texel_x']), int(r['w'])
        if (x % 256) + w > 256: print(rid, 'crosses a tpage edge'); problems += 1; continue
        out, pc = draw(px, clut, r, text)
        if png:
            rgb = np.array([[(c & 31) * 8, (c >> 5 & 31) * 8, (c >> 10 & 31) * 8] for c in pc], np.uint8)
            Image.fromarray(rgb[out]).resize((w * 3, int(r['h']) * 3), Image.NEAREST).save(
                f'private/re/menus/en/{rid}.png')
    problems += build_stats(tl, px, clut, mg, rows, png)
    problems += build_loc(tl, px, clut, mg, png)
    problems += build_map(tl, mg, png)
    raw = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8)
    d[PIX:PIX + 131072] = raw.tobytes()
    os.makedirs('private/build', exist_ok=True)
    open('private/build/MA_COCK.BIZ.lz', 'wb').write(d)
    c = lz.compress(bytes(d))
    open('private/build/MA_COCK.BIZ', 'wb').write(c)
    open('private/build/MGAME.BIN', 'wb').write(mg)
    n_fin = sync_mgfin(mg, regions)
    print(f'MGFIN: {n_fin} tables copied from MGAME')
    print(f'MA_COCK: {len(tl)} labels, compressed {len(c)} bytes, problems {problems}')

if __name__ == '__main__':
    build('--png' in sys.argv)
