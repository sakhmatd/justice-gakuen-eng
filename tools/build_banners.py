# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Repaint the diary-mode event banners (DAT2/MA_BETA.BIZ entries 011-025)
in English.

Reads tl/BANNERS.txt (ID: EN) and the text rectangles in
data/banners.tsv. Writes private/build/MA_BETA.BIZ with the changed
entries recompressed in their original slots. --png writes previews to
private/re/banner/en/.

Each entry is one LZ'd 8bpp TIM, 512x208, with its own 256-colour CLUT
(CLUT at +0x14, pixels at +0x220). The text is painted into the picture.
For each rectangle the builder picks the palette indices that are much
more common inside it than in a ring around it (the text fill and
outline), paints those pixels out by diffusion from the surrounding
background, draws the English with the original fill and outline colours,
and maps the result back to the nearest CLUT colours.
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import lz

ARC = 'private/disc/DAT2/MA_BETA.BIZ'
ENT = 'private/unpacked/MA_BETA.BIZ/%03d.lz'
FONT_T = '/usr/share/fonts/TTF/ariblk.ttf'
FONT_D = '/usr/share/fonts/TTF/arialbd.ttf'
VIS = (16, 496)                         # visible x range

def load_tl(path='tl/BANNERS.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

def rects():
    r = {}
    for line in open('data/banners.tsv', encoding='utf-8'):
        if line.startswith(('#', 'id')): continue
        f = line.rstrip('\n').split('\t')
        if len(f) >= 10 and f[0].startswith('beta0'):
            r[f[0]] = tuple(int(x) for x in f[6:10])
            cols = [np.array([float(v) for v in c.split(',')]) if c.strip() else None
                    for c in (f[12:15] + ['', '', ''])[:3]]
            COLS[f[0]] = cols
    return r

COLS = {}                               # id -> [fill, outline, shadow] (8-bit RGB)

def colour_mask(img, rect, cols, tol=60):
    """Month cards: pixels in the rectangle near the measured fill, outline
    or shadow colour, grown to cover antialiased edges."""
    x0, y0, x1, y1 = rect
    mk = np.zeros(img.shape[:2], bool)
    sub = img[y0:y1, x0:x1]
    near = np.zeros(sub.shape[:2], bool)
    for c in cols:
        if c is not None: near |= np.sqrt(((sub - c) ** 2).sum(-1)) < tol
    mk[y0:y1, x0:x1] = near
    mk = ndimage.binary_closing(mk, iterations=2)
    rect_m = np.zeros_like(mk); rect_m[y0:y1, x0:x1] = True
    return ndimage.binary_dilation(mk, iterations=2) & ndimage.binary_dilation(rect_m, iterations=1)

def rgb15(v): return np.array([(v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3], float)

def text_mask(img, x0, y0, x1, y1):
    """Text pixels inside the rectangle: those far from a background
    estimate diffused in from the rectangle's border. Returns the mask and
    the fill and outline colours (stroke centres and stroke edges)."""
    rect = np.zeros(img.shape[:2], bool); rect[y0:y1, x0:x1] = True
    bg = inpaint(img, rect, iters=600)
    dist = np.sqrt(((img - bg) ** 2).sum(-1))
    mk = rect & (dist > 40)
    mk = ndimage.binary_closing(mk, iterations=1)
    fill, line = colours(img, mk)
    raw = mk
    mk = ndimage.binary_dilation(mk, iterations=2) & ndimage.binary_dilation(rect, iterations=2)
    return mk, fill, line, raw

def colours(img, mk):
    """Fill colour (stroke centres) and outline colour (stroke edges)."""
    core = ndimage.distance_transform_edt(mk)
    fill = np.median(img[core >= 3], axis=0) if (core >= 3).any() else np.median(img[mk], axis=0)
    edge = mk & (core <= 1.5)
    line = np.median(img[edge], axis=0) if edge.any() else fill * 0.3
    return fill, line

def inpaint(img, mk, iters=400):
    """Fill masked pixels by repeated averaging of their neighbours."""
    out = img.copy()
    ys, xs = np.where(mk)
    if not len(ys): return out
    y0, y1, x0, x1 = ys.min() - 3, ys.max() + 4, xs.min() - 3, xs.max() + 4
    y0, x0 = max(0, y0), max(0, x0)
    sub = out[y0:y1, x0:x1]; m = mk[y0:y1, x0:x1]
    # start from the mean of the unmasked border
    for c in range(3):
        ch = sub[..., c]; ch[m] = ch[~m].mean() if (~m).any() else 128
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float) / 4
    for _ in range(iters):
        for c in range(3):
            ch = sub[..., c]
            avg = ndimage.convolve(ch, k, mode='nearest')
            ch[m] = avg[m]
    out[y0:y1, x0:x1] = sub
    return out

def fit(text, h, outline, font, maxw):
    """Largest font size <= h whose ink (outline included) fits maxw.
    Returns (font, bbox, ink width)."""
    size = int(h * 1.0)
    while True:
        f = ImageFont.truetype(font, size)
        b = f.getbbox(text)
        if b[2] - b[0] + 2 * outline <= maxw or size <= 14: break
        size -= 1
    return f, b, b[2] - b[0] + 2 * outline

def draw_text(img, text, cx, cy, h, fill, line, outline, font, maxw, vis=VIS, left=None,
              shadow=None, soff=(0, 0)):
    """Draw centred on cx (ink centre) or with the ink's left edge at left.
    Returns the ink x range (outline included, shadow excluded)."""
    f, b, _ = fit(text, h, outline, font, maxw)
    W, H = b[2] - b[0] + 2 * outline + 4, b[3] - b[1] + 2 * outline + 4
    a = Image.new('L', (W, H), 0)
    ImageDraw.Draw(a).text((outline + 2 - b[0], outline + 2 - b[1]), text, font=f, fill=255)
    a = np.asarray(a) / 255.0
    o = ndimage.grey_dilation(a, footprint=ndimage.generate_binary_structure(2, 1), size=None)
    for _ in range(outline - 1):
        o = ndimage.grey_dilation(o, footprint=ndimage.generate_binary_structure(2, 1))
    x = int(round(cx - W / 2)) if left is None else int(round(left)) - 2
    y = int(round(cy - H / 2))
    x = min(max(x, vis[0]), vis[1] - W)
    y = max(0, y)
    if shadow is not None:
        sx, sy = soff
        sreg = img[y + sy:y + sy + H, x + sx:x + sx + W]
        so = o[:sreg.shape[0], :sreg.shape[1]]
        sreg[:] = sreg * (1 - so[..., None]) + shadow * so[..., None]
    reg = img[y:y + H, x:x + W]
    o, a = o[:reg.shape[0], :reg.shape[1]], a[:reg.shape[0], :reg.shape[1]]
    reg[:] = reg * (1 - o[..., None]) + line * o[..., None]
    reg[:] = reg * (1 - a[..., None]) + fill * a[..., None]
    cols = np.where(o.max(0) > 0.5)[0]
    return (x + cols[0], x + cols[-1] + 1) if len(cols) else (x, x)

def quantize(img, pal, idx):
    """Nearest CLUT colour for every pixel that changed."""
    out = idx.copy()
    ch = np.abs(img - pal[idx]).sum(-1) > 0.5
    flat = img[ch]
    for k in range(0, len(flat), 4096):
        d = ((flat[k:k + 4096, None, :] - pal[None, :, :]) ** 2).sum(-1)
        flat_idx = d.argmin(1)
        sel = np.where(ch.ravel())[0][k:k + 4096]
        out.ravel()[sel] = flat_idx
    return out

# Horizontal layout. The banner is shown 1:1 at screen (0,112) with x 0-15
# and 496-511 hidden, so its visible centre is x 255.5. The month card panel
# (inside the marble frame) is x 26-485, also centred on 255.5. The JP
# original centres each card's text block on that line: the exam titles on
# their own, the date + title pair as one staggered block (date upper left,
# title lower right), and the day + event schedule as two columns.
CX = 255.5
MAXW_B = 448                            # banner text: 16 px margin inside x 16-495
BLOCK_B = 432                           # date + title block width limit
STAG_B = 96                              # date + title stagger limit
MAXW_C = 432                            # month card: 14 px margin inside the panel
INK = {}                                # key -> drawn ink x range (for checks)

def banner_layout(e, jobs, rc):
    """Set cx or left on the banner jobs: a lone title is centred; a date
    and title keep the JP stagger (title left minus date left), reduced
    until the block fits MAXW_B, and the block is centred."""
    for j in jobs:
        j['w'] = fit(j['text'], j['h'], j['ol'], j['font'], j['maxw'])[2]
    if len(jobs) == 1:
        jobs[0]['cx'] = CX; return
    d, t = sorted(jobs, key=lambda j: not j['key'].endswith('date'))
    s = rc[t['key']][0] - rc[d['key']][0]
    s = max(0, min(s, STAG_B, BLOCK_B - t['w']))
    bw = max(d['w'], s + t['w'])
    left = CX - bw / 2
    d['left'], t['left'] = left, left + s

def card_layout(e, tl, rc):
    """Month card jobs. The month name is centred on the panel (its drop
    shadow, 4 px right, counted in). The schedule is a day column (each day
    centred in it) and a left-aligned event column, with the JP gap between
    them; all days share one font size and all events another, and the
    block is centred on the panel (right half of the TIM, x + 512)."""
    jobs = []
    k = f'beta{e:03d}.month'
    if k in tl:
        x0, y0, x1, y1 = rc[k]; fill, line, shadow = COLS[k]
        jobs.append(dict(key=k, text=tl[k], cx=CX - 2, cy=(y0 + y1) / 2, h=(y1 - y0) * 0.5,
                         fill=fill, line=line, ol=3, font=FONT_T, vis=(0, 512), maxw=MAXW_C,
                         shadow=shadow, soff=(4, 6)))
    n = 1; days = []; evs = []
    while f'beta{e:03d}.day{n}' in tl and f'beta{e:03d}.ev{n}' in tl:
        for kind, lst in (('day', days), ('ev', evs)):
            k = f'beta{e:03d}.{kind}{n}'
            x0, y0, x1, y1 = rc[k]; fill, line, shadow = COLS[k]
            lst.append(dict(key=k, text=tl[k], cy=(y0 + y1) / 2, h=(y1 - y0) * 0.68,
                            fill=fill, line=line, ol=1, font=FONT_D, vis=(512, 1024),
                            shadow=shadow, soff=(1, 2)))
        n += 1
    if not days: return jobs
    gap = min(rc[v['key']][0] - rc[d['key']][2] for d, v in zip(days, evs))
    def size(col, maxw):
        return min(fit(j['text'], j['h'], 1, FONT_D, maxw)[0].size for j in col)
    hd = size(days, MAXW_C)
    dw = max(fit(j['text'], hd, 1, FONT_D, MAXW_C)[2] for j in days)
    he = size(evs, MAXW_C - dw - gap)
    ew = max(fit(j['text'], he, 1, FONT_D, MAXW_C)[2] for j in evs)
    left = 512 + CX - (dw + gap + ew) / 2
    for d, v in zip(days, evs):
        d.update(h=hd, maxw=MAXW_C, cx=left + dw / 2)
        v.update(h=he, maxw=MAXW_C, left=left + dw + gap)
    return jobs + days + evs

def build(png=False, out='private/build/MA_BETA.BIZ', pngdir='private/re/banner/en'):
    tl = load_tl(); rc = rects()
    arc = bytearray(open(ARC, 'rb').read())
    toc = [struct.unpack_from('<II', arc, 8 * i) for i in range(26)]
    problems = 0
    if png: os.makedirs(pngdir, exist_ok=True)
    for e in range(0, 26):
        card = e < 11
        if card:
            keys = [k for k in rc if k.startswith(f'beta{e:03d}.')]
            if not any(k in tl for k in keys): continue
        else:
            keys = [k for k in (f'beta{e:03d}.{n}' for n in ('date', 'title')) if k in tl and k in rc]
            if not keys: continue
        W, H = (1024, 256) if card else (512, 208)
        d = bytearray(open(ENT % e, 'rb').read())
        clut = struct.unpack_from('<256H', d, 0x14)
        pal = np.array([rgb15(v) for v in clut])
        idx = np.frombuffer(bytes(d[0x220:0x220 + W * H]), np.uint8).reshape(H, W).copy()
        img = pal[idx]
        jobs = []
        if card:
            for k in keys:
                mk = colour_mask(img, rc[k], COLS[k])
                img = inpaint(img, mk)
            jobs += card_layout(e, tl, rc)
            keys = []
        for k in keys:
            x0, y0, x1, y1 = rc[k]
            mk, fill, line, _ = text_mask(img, x0, y0, x1, y1)
            img = inpaint(img, mk)
            big = k.endswith('title')
            jobs.append(dict(key=k, text=tl[k], cy=(y0 + y1) / 2, h=(y1 - y0) * (0.62 if big else 0.75),
                             fill=fill, line=line, ol=3 if big else 2, font=FONT_T if big else FONT_D,
                             vis=VIS, maxw=MAXW_B, shadow=None, soff=(0, 0)))
        if not card: banner_layout(e, jobs, rc)
        for j in jobs:
            ink = draw_text(img, j['text'], j.get('cx', 0), j['cy'], j['h'], j['fill'], j['line'],
                            j['ol'], j['font'], j['maxw'], j['vis'], j.get('left'),
                            j['shadow'], j['soff'])
            INK[j['key']] = tuple(int(v) for v in ink)
        new = quantize(np.clip(img, 0, 255), pal, idx)
        d[0x220:0x220 + W * H] = new.tobytes()
        c = lz.compress(bytes(d))
        o = toc[e][0]; slot = (toc[e + 1][0] if e + 1 < len(toc) else len(arc)) - o
        if len(c) > slot: print(f'MA_BETA {e:03d}: {len(c)} > slot {slot}'); problems += 1; continue
        arc[o:o + slot] = c + bytes(slot - len(c))
        struct.pack_into('<I', arc, 8 * e + 4, len(c))
        if png: Image.fromarray(pal[new].astype(np.uint8)).save(f'{pngdir}/{e:03d}.png')
    if out: open(out, 'wb').write(arc)
    print(f'MA_BETA: problems {problems}')

if __name__ == '__main__':
    build('--png' in sys.argv)
