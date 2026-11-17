# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Redraw the diary-mode staff roll in English.

Reads tl/STAFF.txt (ID: JP | EN, ID = piece index in STAFF.BIN).
Writes private/build/STAFF.BIN (new piece rectangles and line offsets) and
private/build/MGSTAFF1.BIZ (all 20 entries with the new credit atlas,
recompressed in their original slots). --png writes previews to
private/re/staff/en/. --check builds everything but writes nothing to
private/build/.

STAFF.BIN (DAT1, loads at 0x8009c000):
- 0x594: 153 pointers to the schedule records (8 bytes: mode, line, clut,
  piece, y/10, 0, 0, 0). Mode 0 centres the line on the 512 px screen,
  mode 1 on the right half, mode 2 on the left half; the piece slides in
  from that side. A record with line 0 starts a new page.
- 0x1178: 153 pointers to the piece records (8 s16: tpage x, tpage y,
  u, v, w, h, line width, x offset in the line).
MGSTAFF1.BIZ entry (any of 20, loaded to 0x800a4000): 16 picture frames,
then the credit atlas: TIM A 4bpp 1024x256 at VRAM (768,0), pixels at
0xa0040; TIM B 4bpp 768x256 at VRAM (768,256), pixels at 0xc0920; TIM C
(PRESENTED BY / CAPCOM, VRAM 640,0) is left as is. All pieces are 20 px
rows. Job titles use the CLUT ramp 1-6 (the record's CLUT picks green,
blue or orange), names the grey ramp 8-15; 0 and 7 are transparent.
"""
import sys, os, re, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import lz

STAFF = 'private/disc/DAT1/STAFF.BIN'
ARC = 'private/disc/DAT2/MGSTAFF1.BIZ'
ENT = 'private/unpacked/MGSTAFF1.BIZ/%03d.lz'
FONT = '/usr/share/fonts/TTF/bahnschrift.ttf'
SYMBOL = '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf'
BASE = 0x8009c000
NREC = 153
SIZE, WEIGHT = 19, 600         # font px, Bahnschrift weight axis
MAXW = 240                     # widest line (the original widest is 240)
GAP = 12                       # px between the two pieces of a line
ROW = 20
# atlas pages: (tpage x, tpage y) -> (TIM pixel offset, TIM width, column, bg)
PAGES = [((768, 0), 0), ((832, 0), 0), ((896, 0), 0), ((960, 0), 0),
         ((768, 256), 1), ((832, 256), 1), ((896, 256), 1)]
TIMS = [(0xa0040, 1024, 768, 0), (0xc0920, 768, 768, 7)]
CLUT = 0xc0800 + 20            # TIM B CLUTs (the last ones uploaded)
DITHER = np.array([[0, .5], [.75, .25]])

def load_tl(path='tl/STAFF.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        k, v = line.split(':', 1)
        jp, en = (s.strip() for s in v.split('|'))
        if en == '=': en = jp.replace('・', '{dot}')
        out[int(k)] = (jp, en.replace('{dot}', '·').replace('{star}', '☆'))
    return out

def tables(d):
    sched = [struct.unpack_from('<I', d, 0x594 + 4 * i)[0] - BASE for i in range(NREC)]
    rects = [struct.unpack_from('<I', d, 0x1178 + 4 * i)[0] - BASE for i in range(NREC)]
    return [tuple(d[p:p + 5]) for p in sched], rects

def atlas(ent):
    """TIM A and B as index arrays (256 x W)."""
    out = []
    for pix, W, _, _ in TIMS:
        raw = np.frombuffer(ent[pix:pix + W * 128], np.uint8).reshape(256, W // 2)
        p = np.empty((256, W), np.uint8); p[:, 0::2] = raw & 15; p[:, 1::2] = raw >> 4
        out.append(p)
    return out

def orig_piece(tims, r):
    tx, ty, u, v, w, h = r[:6]
    t = 0 if ty == 0 else 1
    if tx < 768: return None
    u0 = (tx - 768) * 4 + u
    return tims[t][v:v + h, u0:u0 + w]

def font(size, width):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_axes([WEIGHT, width])
    return f

SETTINGS = [(SIZE, w) for w in range(100, 74, -5)] + [(s, 75) for s in range(SIZE - 1, 13, -1)]

def render(text, maxw=MAXW, start=0):
    """Alpha (20 x w) for text. Walks SETTINGS (narrower font, then smaller)
    from index start until the text fits maxw. Returns (alpha, setting index)."""
    ss = 4
    for k in range(start, len(SETTINGS)):
        size, width = SETTINGS[k]
        f = font(size * ss, width)
        fs = ImageFont.truetype(SYMBOL, size * ss * 4 // 5)
        # runs: Bahnschrift has no white star, DejaVu draws it
        runs = [(t, fs if t == '\u2606' else f) for t in re.split('(\u2606)', text) if t]
        tw = sum(ff.getlength(t) for t, ff in runs)
        if tw <= maxw * ss: break
    w = int(tw + ss - 1) // ss + 2
    im = Image.new('L', (w * ss, ROW * ss), 0)
    cap = f.getbbox('H')
    top = (2 * ss) - cap[1] + (SIZE - size) * ss // 2
    dr = ImageDraw.Draw(im); x = 0
    for t, ff in runs:
        if ff is f: dr.text((x, top), t, font=f, fill=255)
        else:
            sb = ff.getbbox(t)
            dr.text((x, top + (cap[1] + cap[3]) // 2 - (sb[1] + sb[3]) // 2), t, font=ff, fill=255)
        x += ff.getlength(t)
    a = np.asarray(im.resize((w, ROW), Image.LANCZOS), float) / 255
    a[a < 0.12] = 0
    cols = np.nonzero(a.max(0))[0]
    a = a[:, cols[0]:cols[-1] + 1] if len(cols) else a[:, :1]
    return a, k

def to_index(a, title):
    t = DITHER[np.arange(a.shape[0])[:, None] % 2, np.arange(a.shape[1])[None, :] % 2]
    if title: idx = 1 + np.floor(a * 4.9 + t)
    else: idx = 8 + np.floor(a * 5.5 + t * 0.67)
    idx = np.clip(idx, 1 if title else 8, 6 if title else 14).astype(np.uint8)
    idx[a <= 0] = 0
    return idx

def pages_of(sched):
    pages, cur = [], None
    for i, (m, b, c, pid, y) in enumerate(sched):
        if b == 0 and not (cur and m == 0 and cur[0][1][0] == 0): cur = []; pages.append(cur)
        cur.append((i, sched[i]))
    return pages

def layout(sched, tl, alpha, fit, kinds):
    """Line width and x offset per piece. The two-piece lines of a page with
    the same kinds (title+name or name+name) form two aligned columns (title
    right-aligned, name left-aligned). If the columns are too wide, the whole
    group is re-rendered with one narrower setting."""
    pages = pages_of(sched)
    place = {}
    for pg in pages:
        lines = {}
        for i, (m, b, c, pid, y) in pg:
            if pid in alpha: lines.setdefault(y, []).append(pid)
        pairs = {}
        for y, ps in lines.items():
            if len(ps) == 1:
                place[ps[0]] = (alpha[ps[0]].shape[1], 0)
            else:
                assert len(ps) == 2, ps
                pairs.setdefault(tuple(kinds[p] for p in ps), []).append(ps)
        for kk, group in pairs.items():
            ids = [p for pq in group for p in pq]
            k = max(fit[p] for p in ids)
            while True:
                for p in ids:
                    if fit[p] != k: alpha[p], fit[p] = render(tl[p][1], 999, k)
                c1 = max(alpha[p].shape[1] for p, q in group)
                c2 = max(alpha[q].shape[1] for p, q in group)
                if c1 + GAP + c2 <= MAXW or k == len(SETTINGS) - 1: break
                k += 1
            for p, q in group:
                w1 = alpha[p].shape[1]
                place[p] = (c1 + GAP + c2, c1 - w1 if kk[0] == 'title' else 0)
                place[q] = (c1 + GAP + c2, c1 + GAP)
    return place, pages

def pack(widths):
    """First-fit decreasing into 7 pages x 12 rows of 256 texels."""
    rows = [[pg, r, 0] for pg in range(len(PAGES)) for r in range(256 // ROW)]
    at = {}
    for pid in sorted(widths, key=lambda p: -widths[p]):
        w = widths[pid]
        for row in rows:
            if row[2] + w <= 256:
                at[pid] = (row[0], row[1] * ROW, row[2]); row[2] += w; break
        else:
            raise SystemExit(f'atlas full at piece {pid} ({w} px)')
    used = sum(r[2] for r in rows)
    return at, used

def build(png=False, check=False):
    tl = load_tl()
    sd = bytearray(open(STAFF, 'rb').read())
    sched, rptr = tables(sd)
    rects = [struct.unpack_from('<8h', sd, p) for p in rptr]
    ent0 = open(ENT % 0, 'rb').read()
    tims = atlas(ent0)
    used_ids = sorted({s[3] for s in sched})
    alpha, kinds, fit = {}, {}, {}
    for pid in used_ids:
        r = rects[pid]
        if r[0] < 768: continue                       # TIM C, already English
        op = orig_piece(tims, r)
        kinds[pid] = 'title' if ((op >= 1) & (op <= 6)).sum() > ((op >= 8)).sum() else 'name'
        if pid not in tl: raise SystemExit(f'piece {pid} missing in STAFF.txt')
        centred = all(m == 0 for m, b, c, q, y in sched if q == pid)
        alpha[pid], fit[pid] = render(tl[pid][1], 256 if centred else MAXW)
    place, pages = layout(sched, tl, alpha, fit, kinds)
    problems = 0
    for pid, (tot, off) in place.items():
        if tot > 256: print(f'piece {pid}: line {tot} px > 256'); problems += 1
    at, used = pack({p: a.shape[1] for p, a in alpha.items()})
    # new atlas
    new = [np.full_like(t, bg) for t, (_, _, _, bg) in zip(tims, TIMS)]
    for pid, (pg, v, u) in at.items():
        (tx, ty), t = PAGES[pg]
        idx = to_index(alpha[pid], kinds[pid] == 'title')
        x0 = (tx - 768) * 4 + u
        sub = new[t][v:v + ROW, x0:x0 + idx.shape[1]]
        sub[idx != 0] = idx[idx != 0]
        tot, off = place[pid]
        struct.pack_into('<8h', sd, rptr[pid], tx, ty, u, v, idx.shape[1], ROW, tot, off)
    # entries
    arc = bytearray(open(ARC, 'rb').read())
    toc = [struct.unpack_from('<II', arc, 8 * i) for i in range(20)]
    packed = [(n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8).tobytes() for n in new]
    sizes = []
    for e in range(20):
        d = bytearray(open(ENT % e, 'rb').read())
        for (pix, W, _, _), pb in zip(TIMS, packed): d[pix:pix + W * 128] = pb
        c = lz.compress(bytes(d))
        o = toc[e][0]
        slot = (toc[e + 1][0] if e < 19 else len(arc)) - o
        if len(c) > slot: print(f'MGSTAFF1 {e:03d}: {len(c)} > slot {slot}'); problems += 1; continue
        arc[o:o + slot] = c + bytes(slot - len(c))
        struct.pack_into('<I', arc, 8 * e + 4, len(c))
        sizes.append(len(c))
    if png: preview(sched, rects_after(sd, rptr), new, ent0, pages, tl)
    print(f'STAFF: {len(alpha)} pieces, atlas {used} of {len(PAGES) * 12 * 256} texels used, '
          f'entries {min(sizes):#x}-{max(sizes):#x} bytes, problems {problems}')
    narrowed = {p: SETTINGS[k] for p, k in fit.items() if k}
    if narrowed: print('narrowed (size/width%):', ' '.join(f'{p}:{s}/{w}' for p, (s, w) in sorted(narrowed.items())))
    if problems or check: return problems
    open('private/build/STAFF.BIN', 'wb').write(sd)
    open('private/build/MGSTAFF1.BIZ', 'wb').write(arc)
    return 0

def rects_after(sd, rptr):
    return [struct.unpack_from('<8h', sd, p) for p in rptr]

def preview(sched, rects, new, ent, pages, tl):
    """Static pages as STAFF.BIN places them (no slide-in), plus the atlas."""
    os.makedirs('private/re/staff/en', exist_ok=True)
    cl = struct.unpack_from('<128H', ent, CLUT)
    pal = lambda c: np.array([((v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3)
                              for v in cl[c * 16:c * 16 + 16]], np.uint8)
    rawc = np.frombuffer(ent[0xd9120:0xd9120 + 128 * 256], np.uint8).reshape(256, 128)
    timc = np.empty((256, 256), np.uint8); timc[:, 0::2] = rawc & 15; timc[:, 1::2] = rawc >> 4
    for k, pg in enumerate(pages):
        H = max(s[4] for _, s in pg) * 10 + 30
        im = Image.new('RGB', (512, max(H, 480)), (24, 24, 32))
        dr = ImageDraw.Draw(im)
        dr.line([(256, 0), (256, im.height)], fill=(60, 60, 60))
        for i, (m, b, c, pid, y) in pg:
            tx, ty, u, v, w, h, tot, off = rects[pid]
            if tx < 768: src = timc[v:v + h, (tx - 640) * 4 + u:][:, :w]
            else: src = new[0 if ty == 0 else 1][v:v + h, (tx - 768) * 4 + u:][:, :w]
            x = {0: (512 - tot) // 2, 1: 256 + (256 - tot) // 2, 2: (256 - tot) // 2}[m] + off
            mask = Image.fromarray(((src != 0) & (src != 7)).astype(np.uint8) * 255)
            im.paste(Image.fromarray(pal(c)[src]), (x, y * 10), mask)
        im.save(f'private/re/staff/en/page{k:02d}.png')
    for t, n in enumerate(new):
        Image.fromarray(pal(0)[n]).save(f'private/re/staff/en/atlas_{"ab"[t]}.png')

if __name__ == '__main__':
    sys.exit(build('--png' in sys.argv, '--check' in sys.argv))
