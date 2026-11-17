# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Repaint the Festival minigame pictures (DAT2/MINITTL1-8.BIZ) and the
CEFT5 HUD labels in English.

Run after build_fest.py. Reads tl/FESTPIC.txt and the rectangles and
styles in data/fest_pics.tsv (RE notes: private/re/fest/pics_notes.txt).

- Title pictures (TIM 0x18800, t1-t8): the JP title, the speech bubble text
  and the 練習用 stamp are painted out and redrawn in English.
- Rules pages (TIM 0x55000, and 0x91800 in MINITTL2/6): band header,
  section headings on their ellipse plates, body text, labels, grey label
  plates and the 戻る hint. MINITTL5-8 rules pages are byte-identical to
  1-4, so each is painted once and copied.
- CEFT5.BIZ HUD atlas (TIM 0x10800, 4bpp): the short labels are redrawn in
  place within their sprite rects. The end-of-round box (Play again? /
  Yes / No, rect 656,168 112x72, CLUT 49) is erased to its fill stipple
  and redrawn (AGAIN, paint_again; RE: private/re/festprac/findings-A2.org).

The picture TIMs are always taken from private/unpacked (originals), so the
tool can be rerun. The rest of each MINITTL file comes from
private/build/MINITTLn.BIZ.lz when it exists (atlas edits of build_fest.py).
Writes private/build/MINITTLn.BIZ(.lz) and CEFT5.BIZ(.lz). --png writes
before/after previews to private/re/fest/en/. --dry skips compression and
writing. Page tags (r1 r2a r2b r3 r4 t1..t8 hud) limit the run to those
pages (for previews; a real build should paint everything).
FESTPICS_ERASE_ONLY=1 stops the titles after the erase step (debugging).
"""
import sys, os, struct, colorsys, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import lz

FD = '/usr/share/fonts/TTF/'
F_BODY = FD + 'tahomabd.ttf'
F_HEAD = FD + 'seguibli.ttf'
F_IMPACT = FD + 'impact.ttf'
F_SERIF = FD + 'georgiaz.ttf'
F_SCRIPT = FD + 'segoescb.ttf'
F_STAMP = FD + 'ariblk.ttf'
TITLE, RULES, RULES2 = 0x18800, 0x55000, 0x91800
PW, PH = 512, 480
PREV = 'private/re/fest/en'
RNG = np.random.default_rng(1)

# ---------------------------------------------------------------- data

def load_tl(path='tl/FESTPIC.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'): continue
        k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

def rgb_of(s):
    m = re.findall(r'(\d+),(\d+),(\d+)', s)
    return [np.array([float(a), float(b), float(c)]) for a, b, c in m]

def load_rows(path='data/fest_pics.tsv'):
    rows = {}
    for line in open(path, encoding='utf-8'):
        if line.startswith(('#', 'id\t')) or not line.strip(): continue
        f = line.rstrip('\n').split('\t')
        r = dict(id=f[0], file=f[1], off=int(f[2], 16), x=int(f[3]), y=int(f[4]),
                 w=int(f[5]), h=int(f[6]), kind=f[7], fill=f[8], outline=f[9],
                 shadow=f[10], gh=int(f[11]), lines=int(f[12]), jp=f[13],
                 notes=f[14] if len(f) > 14 else '')
        r['n'] = int(f[1][7])
        for key in ('num', 'icon', 'botan', 'plate'):
            r[key] = [tuple(int(v) for v in m.split(','))
                      for m in re.findall(key + r'@(\d+,\d+,\d+,\d+)', r['notes'])]
        sh = r['shadow']
        m = re.match(r'(\d+),(\d+),(\d+)/\+(\d+)\+(\d+)', sh)
        r['sh_col'] = np.array([float(m[1]), float(m[2]), float(m[3])]) if m else None
        r['sh_off'] = (int(m[4]), int(m[5])) if m else (1, 2)
        rows[r['id']] = r
    return rows

def rgb15(v): return np.array([(v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3], float)

class Page:
    """One 8bpp 512x480 picture with its 256-colour CLUT."""
    def __init__(self, src, off):
        self.off = off
        self.clut = list(struct.unpack_from('<256H', src, off + 0x14))
        self.pal = np.array([rgb15(v) for v in self.clut])
        self.idx = np.frombuffer(bytes(src[off + 0x220:off + 0x220 + PW * PH]),
                                 np.uint8).reshape(PH, PW).copy()
        self.img = self.pal[self.idx]
        self.orig = self.img.copy()

    def store(self, d):
        new = quantize(np.clip(self.img, 0, 255), self.pal, self.idx)
        d[self.off + 0x220:self.off + 0x220 + PW * PH] = new.tobytes()
        self.idx = new
        return self.pal[new]

def quantize(img, pal, idx):
    """Nearest CLUT colour for every pixel that changed (weighted RGB)."""
    out = idx.copy()
    ch = np.abs(img - pal[idx]).sum(-1) > 3
    flat = img[ch]; wt = np.array([0.30, 0.59, 0.11]) ** 0.5
    pw = pal * wt; sel_all = np.where(ch.ravel())[0]
    for k in range(0, len(flat), 4096):
        q = flat[k:k + 4096] * wt
        dd = ((q[:, None, :] - pw[None]) ** 2).sum(-1)
        out.ravel()[sel_all[k:k + 4096]] = dd.argmin(1)
    return out

# ---------------------------------------------------------------- inpaint

def inpaint(img, mk, domain=None, iters=300):
    """Fill mk by repeated averaging of the 4 neighbours inside domain.
    Large holes start from a half-resolution solution."""
    out = img.copy()
    if not mk.any(): return out
    if domain is None: domain = np.ones(mk.shape, bool)
    ys, xs = np.where(mk)
    y0, y1 = max(0, ys.min() - 4), min(mk.shape[0], ys.max() + 5)
    x0, x1 = max(0, xs.min() - 4), min(mk.shape[1], xs.max() + 5)
    sub = out[y0:y1, x0:x1].copy(); m = mk[y0:y1, x0:x1]; dm = domain[y0:y1, x0:x1] | m
    known = dm & ~m
    if not known.any(): return out
    _solve(sub, m, dm, known, iters)
    out[y0:y1, x0:x1] = sub
    return out

def _solve(sub, m, dm, known, iters):
    h, w = m.shape
    if min(h, w) > 16 and ndimage.distance_transform_edt(m).max() > 6:
        # coarse start
        hs, ws = (h + 1) // 2, (w + 1) // 2
        def down(a, f):
            p = np.zeros((hs * 2, ws * 2) + a.shape[2:], a.dtype); p[:h, :w] = a
            return f(p.reshape(hs, 2, ws, 2, *a.shape[2:]))
        kc = down(known.astype(float), lambda p: p.sum((1, 3)))
        sc = down(sub * known[..., None], lambda p: p.sum((1, 3)))
        dc = down(dm, lambda p: p.any((1, 3)))
        csub = sc / np.maximum(kc, 1)[..., None]
        cknown = kc > 0
        cm = dc & ~cknown
        if cm.any() and cknown.any():
            _solve(csub, cm, dc, cknown, iters)
        up = np.repeat(np.repeat(csub, 2, 0), 2, 1)[:h, :w]
        sub[m] = up[m]
    else:
        for c in range(3):
            sub[..., c][m] = sub[..., c][known].mean()
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    wgt = ndimage.convolve(dm.astype(float), k, mode='constant')
    wgt = np.maximum(wgt, 1e-6)
    for _ in range(iters):
        for c in range(3):
            ch = sub[..., c]
            avg = ndimage.convolve(ch * dm, k, mode='constant') / wgt
            ch[m] = avg[m]

def protect_mask_all(sh, rows, grow=0):
    p = np.zeros(sh, bool)
    for r in rows: p |= protect_mask(sh, r, grow)
    return p

def grain(img, mk, ref, amount=1.0, ignore=None):
    """Add noise with the local standard deviation of the unmasked pixels
    around the hole (the picture's quantisation noise)."""
    hi = ref - ndimage.uniform_filter(ref, size=(3, 3, 1))
    k = np.ones((9, 9))
    bad = mk if ignore is None else mk | ignore
    known = (~ndimage.binary_dilation(bad, iterations=3)).astype(float)
    var = ndimage.convolve((hi ** 2).sum(-1) * known, k) / np.maximum(ndimage.convolve(known, k), 1)
    var = ndimage.grey_dilation(var, size=(5, 5))
    sd = np.sqrt(var / 3) * amount
    out = img.copy()
    n = RNG.standard_normal(img.shape[:2])
    out[mk] += (n[mk] * sd[mk])[:, None]
    return out

def rect_mask(shape, x, y, w, h, grow=0):
    m = np.zeros(shape, bool)
    m[max(0, y - grow):y + h + grow, max(0, x - grow):x + w + grow] = True
    return m

# ---------------------------------------------------------------- text

_fcache = {}
def font(path, size):
    k = (path, size)
    if k not in _fcache: _fcache[k] = ImageFont.truetype(path, size)
    return _fcache[k]

def render(text, path, size, shear=0.0, sx=1.0, ss=4, track=0.0, per_letter=False):
    """Alpha (h, w) of one line of text. Returns (alpha, baseline_y, labels)
    where labels[y, x] is the index of the letter that owns the pixel (or -1)
    when per_letter is set. size is the font size in output pixels."""
    f = font(path, int(round(size * ss)))
    asc, desc = f.getmetrics()
    pad = int(2 * ss + abs(shear) * (asc + desc))
    xs = []; x = 0.0
    for i, ch in enumerate(text):
        xs.append(x)
        x += f.getlength(ch) + track * ss
    # kerning-aware total for plain text
    tw = int(max(x, f.getlength(text)) + 2 * pad)
    th = asc + desc + 2 * pad
    def draw(sub, at):
        im = Image.new('L', (tw, th), 0)
        ImageDraw.Draw(im).text((pad + at, pad), sub, font=f, fill=255)
        if shear:
            im = im.transform(im.size, Image.AFFINE, (1, shear, -shear * (pad + asc), 0, 1, 0),
                              resample=Image.BICUBIC)
        nw = max(1, int(round(tw * sx / ss))); nh = max(1, int(round(th / ss)))
        return np.asarray(im.resize((nw, nh), Image.BOX), float) / 255.0
    if per_letter or track:
        layers = [draw(ch, xs[i]) for i, ch in enumerate(text)]
        a = np.max(layers, axis=0) if layers else draw('', 0)
        lab = np.argmax(layers, axis=0) if layers else np.zeros(a.shape, int)
        lab[a < 0.02] = -1
    else:
        a = draw(text, 0); lab = None
    # crop empty columns/rows but remember the baseline
    cols = np.where(a.max(0) > 0.01)[0]; rows_ = np.where(a.max(1) > 0.01)[0]
    base = (pad + asc) / ss
    if not len(cols): return np.zeros((1, 1)), 0, None
    a2 = a[rows_[0]:rows_[-1] + 1, cols[0]:cols[-1] + 1]
    if lab is not None: lab = lab[rows_[0]:rows_[-1] + 1, cols[0]:cols[-1] + 1]
    return a2, base - rows_[0], lab

def text_width(text, path, size, sx=1.0, track=0.0):
    f = font(path, max(1, int(round(size * 4))))
    return (f.getlength(text) + track * 4 * max(0, len(text) - 1)) * sx / 4

def wrap(text, path, size, width, first_indent=0, sx=1.0):
    lines = []
    for para in text.split('|'):
        cur = ''
        for word in para.split():
            t = (cur + ' ' + word).strip()
            lim = width - (first_indent if not lines else 0)
            if not cur or text_width(t, path, size, sx) <= lim: cur = t
            else: lines.append(cur); cur = word
        lines.append(cur)
    return lines

def fit_block(text, path, w, h, smax, smin, pitch_k=1.35, indent=0, sx_min=0.88):
    """Largest size (and horizontal squeeze) whose wrapped lines fit w x h."""
    for size in np.arange(smax, smin - 0.01, -0.5):
        for sx in (1.0, 0.94, sx_min):
            lines = wrap(text, path, size, w, indent, sx)
            pitch = max(size * pitch_k, size + 2)
            if all(text_width(l, path, size, sx) <= w - (indent if i == 0 else 0)
                   for i, l in enumerate(lines)) and \
               (len(lines) - 1) * pitch + size * 1.0 <= h:
                return size, sx, lines, pitch
    lines = wrap(text, path, smin, w, indent, sx_min)
    return smin, sx_min, lines, max(smin * pitch_k, smin + 2)

def over(img, x, y, a, col, alpha=1.0):
    """Composite colour (RGB or HxWx3 map) at (x, y) with alpha a."""
    h, w = a.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0: return
    aa = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * alpha
    c = col if np.ndim(col) == 1 else col[y0 - y:y1 - y, x0 - x:x1 - x]
    reg = img[y0:y1, x0:x1]
    reg[:] = reg * (1 - aa) + c * aa

def dilate(a, r):
    if r <= 0: return a
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    fp = (xx * xx + yy * yy) <= r * r + 0.5
    p = np.pad(a, r)
    return ndimage.grey_dilation(p, footprint=fp)

# ---------------------------------------------------------------- masks

def ink_mask(img, r, fill=None, tol=90, grow=2, extra=2):
    """Text fill pixels close to the fill colour, plus their drop shadow,
    in the row's rect (grown by `extra`), minus protected icon rects."""
    sh = img.shape[:2]
    R = rect_mask(sh, r['x'], r['y'], r['w'], r['h'], extra)
    if fill is None: fill = rgb_of(r['fill'])[0]
    f = (np.sqrt(((img - fill) ** 2).sum(-1)) < tol) & R
    sdx, sdy = r['sh_off']
    m = f.copy()
    for dy in range(0, sdy + 2):
        for dx in range(0, sdx + 2):
            m |= np.roll(np.roll(f, dy, 0), dx, 1)
    m = ndimage.binary_dilation(m, iterations=grow) & rect_mask(sh, r['x'], r['y'], r['w'], r['h'], extra + 2)
    return m & ~protect_mask(sh, r)

def protect_mask(sh, r, grow=1):
    """Icons and numbered circles are round: protect the inscribed ellipse
    (so commas in the rect corners can be erased); plates are rects."""
    p = np.zeros(sh, bool)
    yy, xx = np.ogrid[0:sh[0], 0:sh[1]]
    for key in ('num', 'icon'):
        for (x, y, w, h) in r[key]:
            # button discs are ~16 px wide; wider icon rects include the
            # 、 that follows the disc on its right
            if key == "icon" and w > 16: w = 15
            cx, cy = x + (w - 1) / 2, y + (h - 1) / 2
            p |= ((xx - cx) / (w / 2 + grow + 0.5)) ** 2 + ((yy - cy) / (h / 2 + grow + 0.5)) ** 2 <= 1
    for (x, y, w, h) in r['plate']:
        p |= rect_mask(sh, x, y, w, h, grow)
    return p

def diff_mask(img, x0, y0, x1, y1, thr=40, grow=2, unknown=None):
    """Pixels in the rect far from a background diffused in from its border
    (pixels in `unknown` do not feed the background estimate)."""
    rect = np.zeros(img.shape[:2], bool); rect[y0:y1, x0:x1] = True
    bg = inpaint(img, rect if unknown is None else rect | unknown, iters=200)
    dist = np.sqrt(((img - bg) ** 2).sum(-1))
    mk = rect & (dist > thr)
    mk = ndimage.binary_closing(mk, iterations=1)
    return ndimage.binary_dilation(mk, iterations=grow) & ndimage.binary_dilation(rect, iterations=grow)

def mode_colour(img, m):
    px = img[m].astype(int)
    if not len(px): return None
    key = px[:, 0] // 8 * 1024 + px[:, 1] // 8 * 32 + px[:, 2] // 8
    v, c = np.unique(key, return_counts=True)
    k = v[c.argmax()]
    return np.median(px[key == k], 0).astype(float)

# ---------------------------------------------------------------- body text

def draw_lines(img, lines, x, y, path, size, sx, pitch, fill, shadow, soff, indent=0,
               align='left', width=None, outline=None):
    """Draw lines with their tops at y + i * pitch. Returns the bounding box."""
    asc = font(path, int(round(size * 4))).getmetrics()[0] / 4
    box = [9999, 9999, 0, 0]
    for i, t in enumerate(lines):
        if not t: continue
        a, base, _ = render(t, path, size, sx=sx)
        ly = int(round(y + i * pitch + (pitch - size) / 2 + asc - base - size * 0.12))
        if align == 'centre':
            lx = int(round(x + (width - a.shape[1]) / 2))
        else:
            lx = x + (indent if i == 0 else 0)
        if shadow is not None:
            over(img, lx + soff[0], ly + soff[1], a, shadow, 0.95)
        if outline is not None:
            o = dilate(a, 1); over(img, lx - 1, ly - 1, o, outline)
        over(img, lx, ly, a, fill)
        box = [min(box[0], lx), min(box[1], ly), max(box[2], lx + a.shape[1]), max(box[3], ly + a.shape[0])]
    return box

# per-row layout overrides: area (x0, y0, x1, y1) the new text may use,
# size limits, vertical alignment
AREA = {
    'r1_rule': dict(area=(40, 122, 482, 208)),
    'r1_c1': dict(area=(52, 274, 278, 336)),
    'r1_c2': dict(area=(51, 354, 278, 410)),
    'r1_k1': dict(area=(370, 306, 486, 334)),
    'r1_k2': dict(area=(370, 352, 486, 380)),
    'r2a_rule': dict(area=(24, 121, 494, 178)),
    'r2a_c1': dict(area=(36, 245, 292, 298)),
    'r2a_c2': dict(area=(35, 306, 292, 362)),
    'r2a_c3': dict(area=(33, 372, 292, 452)),
    'r2a_k1': dict(area=(329, 217, 494, 256)),
    'r2a_k2': dict(area=(331, 260, 494, 284)),
    'r2a_k3': dict(area=(330, 296, 494, 320)),
    'r2a_k4': dict(area=(329, 328, 494, 367)),
    'r2a_k5': dict(area=(339, 374, 494, 396)),
    'r2a_k6': dict(area=(338, 405, 494, 428)),
    'r2b_c1': dict(area=(41, 186, 274, 214)),
    'r2b_c2': dict(area=(40, 234, 274, 342)),
    'r2b_k1': dict(area=(326, 166, 494, 206)),
    'r2b_k2': dict(area=(328, 208, 494, 234)),
    'r2b_k3': dict(area=(329, 244, 494, 268)),
    'r2b_k4': dict(area=(327, 277, 494, 318)),
    'r2b_gauge_t': dict(area=(310, 346, 494, 390)),
    'r3_rule': dict(area=(11, 104, 503, 199), smax=14, smin=11),
    'r3_c1': dict(area=(37, 245, 298, 299)),
    'r3_c2': dict(area=(37, 307, 298, 361)),
    'r3_c3': dict(area=(36, 372, 298, 426)),
    'r3_k1': dict(area=(357, 285, 488, 309)),
    'r3_k2': dict(area=(358, 320, 488, 344)),
    'r3_k3': dict(area=(366, 364, 488, 390)),
    'r4_rule': dict(area=(30, 117, 492, 200)),
    'r4_c1': dict(area=(37, 258, 292, 304)),
    'r4_c2': dict(area=(36, 310, 292, 336)),
    'r4_c3': dict(area=(36, 344, 292, 390)),
    'r4_c4': dict(area=(36, 396, 292, 422)),
    'r4_power_t': dict(area=(325, 296, 494, 334)),
    'r4_angle_t': dict(area=(324, 368, 494, 414)),
}

def body(pg, r, text, jobs):
    a = AREA.get(r['id'], {})
    x0, y0, x1, y1 = a.get('area', (r['x'], r['y'], r['x'] + r['w'], r['y'] + r['h']))
    small = r['gh'] <= 15 or r['kind'] == 'label'
    smax = a.get('smax', 13.5 if small else 16.5)
    smin = a.get('smin', 10 if r['kind'] == 'label' else 11)
    indent = 0
    if text.startswith('{icons}'):
        text = text[len('{icons}'):].strip()
        indent = max(ix + min(iw, 17) for ix, iy, iw, ih in r["icon"]) + 5 - x0
    pk = 1.3 if small else 1.45
    size, sx, lines, pitch = fit_block(text, F_BODY, x1 - x0, y1 - y0, smax, smin, pk, indent)
    if pitch * len(lines) > y1 - y0 + 4:
        pitch = max(size + 1, (y1 - y0) / len(lines))
    if r['id'].endswith('_rule'):
        # free-standing paragraph: centre the block in its box
        y0 = int(round(y0 + max(0, (y1 - y0) - pitch * len(lines)) / 2))
    fill = rgb_of(r['fill'])[0]
    sh = r['sh_col'] if r['sh_col'] is not None else np.array([16., 16, 16])
    jobs.append(lambda: draw_lines(pg.img, lines, x0, y0, F_BODY, size, sx, pitch, fill, sh,
                                   r['sh_off'], indent))
    return f'{r["id"]}: size {size} sx {sx} lines {len(lines)}'

# ---------------------------------------------------------------- rules page

def erase_body(pg, r):
    mk = ink_mask(pg.img, r)
    for (x, y, w, h) in r['botan']:
        mk |= botan_mask(pg.img, x, y, w, h, r)
    return mk

def botan_mask(img, x, y, w, h, r):
    sh = img.shape[:2]
    R = rect_mask(sh, x, y, w, h, 1)
    ring = rect_mask(sh, x, y, w, h, 4) & ~R
    bg = np.median(img[ring & ~protect_mask(sh, r, 0)], 0)
    dist = np.sqrt(((img - bg) ** 2).sum(-1))
    m = R & (dist > 30)
    m = ndimage.binary_dilation(m, iterations=1) & rect_mask(sh, x, y, w, h, 2)
    # keep the icon itself (botan rect touches the icon bottom)
    for (ix, iy, iw, ih) in r['icon'] + r['plate']:
        m &= ~rect_mask(sh, ix, iy, iw, ih - 1, 0)
    return m

def header_band(pg, r, text, jobs):
    img = pg.img
    x, y, w, h = r['x'], r['y'], r['w'], r['h']
    ring = rect_mask(img.shape[:2], x, y, w, h) & ~rect_mask(img.shape[:2], x + 3, y + 3, w - 6, h - 6)
    band = mode_colour(img, ring)
    near = np.sqrt(((img - band) ** 2).sum(-1)) < 28
    # band pixels near the rect (the shadow can cut the band in pieces)
    comp = near & rect_mask(img.shape[:2], x, y, w, h, 8)
    comp = ndimage.binary_opening(comp, iterations=1)
    ys, xs = np.where(comp)
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    dom = np.zeros(img.shape[:2], bool); dom[by0 + 1:by1 - 1, bx0 + 1:bx1 - 1] = True
    mk = dom & ~comp
    mk = ndimage.binary_dilation(mk, iterations=1) & dom
    pg.img = inpaint(img, mk, dom, iters=150)
    # text
    sh_col = r['sh_col']; soff = r['sh_off']
    avail_w = (bx1 - bx0) - 24; avail_h = (by1 - by0) - 8
    size = 34
    while True:
        a, base, lab_ = render(text, F_HEAD, size, per_letter=True)
        sx = min(1.0, avail_w / (a.shape[1] + 2 + soff[0]))
        if (sx >= 0.86 and a.shape[0] + soff[1] + 2 <= avail_h) or size <= 16: break
        size -= 1
    if sx < 1.0:
        a, base, lab_ = render(text, F_HEAD, size, sx=sx, per_letter=True)
    letters = [i for i, c in enumerate(text) if c.strip()]
    hue = {}
    for j, i in enumerate(letters):
        t = j / max(1, len(letters) - 1)
        hue[i] = (1.0 - 0.97 * t) % 1.0
    colmap = np.zeros(a.shape + (3,))
    for i, hh in hue.items():
        rgbc = np.array(colorsys.hsv_to_rgb(hh, 0.95, 1.0)) * 255
        colmap[lab_ == i] = rgbc
    # light top, slightly darker bottom per glyph (as the original)
    shade = np.linspace(1.0, 0.82, a.shape[0])[:, None, None]
    colmap = colmap * shade
    o = dilate(a, 1)
    cx = (bx0 + bx1) / 2; cy = (by0 + by1) / 2
    X = int(round(cx - o.shape[1] / 2 - soff[0] / 2)); Y = int(round(cy - o.shape[0] / 2 - soff[1] / 2))
    def job():
        over(pg.img, X + soff[0], Y + soff[1], o, sh_col)
        over(pg.img, X, Y, o, np.array([248., 248, 248]))
        over(pg.img, X + 1, Y + 1, a, colmap)
    jobs.append(job)
    return f'{r["id"]}: size {size} sx {sx:.2f}'

def section_heading(pg, r, text, jobs):
    img = pg.img; shp = img.shape[:2]
    x, y, w, h = r['x'], r['y'], r['w'], r['h']
    # plate colour: the commonest colour in the rect that is not also in
    # strips above and below it (the background)
    strips = rect_mask(shp, x - 20, y - 10, w + 40, 6) | rect_mask(shp, x - 20, y + h + 4, w + 40, 6)
    key = lambda px: (px[..., 0].astype(int) // 8 * 1024 + px[..., 1].astype(int) // 8 * 32
                      + px[..., 2].astype(int) // 8)
    bgk = set(np.unique(key(img[strips])).tolist())
    rm = rect_mask(shp, x, y, w, h)
    cand = rm & ~np.isin(key(img), list(bgk))
    plate = mode_colour(img, cand)
    win = rect_mask(shp, x - 40, y - 6, w + 80, h + 12)
    near = (np.sqrt(((img - plate) ** 2).sum(-1)) < 40) & win
    near = ndimage.binary_closing(near, structure=np.ones((3, 9)), iterations=2)
    lab, _ = ndimage.label(near)
    k = np.bincount(lab[y:y + h, x:x + w].ravel()); k[0] = 0
    comp = lab == k.argmax()
    ys, xs = np.where(comp)
    ex0, ex1, ey0, ey1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cy = (ey0 + ey1) / 2; b = (ey1 - ey0) / 2
    # erase: ellipse + text
    yy, xx = np.mgrid[0:shp[0], 0:shp[1]]
    ell = ((xx - (ex0 + ex1) / 2) / ((ex1 - ex0) / 2 + 2)) ** 2 + ((yy - cy) / (b + 2)) ** 2 <= 1
    mk = ell | diff_mask(img, x - 2, y - 2, x + w + 4, y + h + 4, thr=45, grow=1)
    mk = ndimage.binary_dilation(mk, iterations=1)
    new = inpaint(img, mk, iters=250)
    pg.img = grain(new, mk, img, 0.8)
    # radial colour profile of the old plate (centre -> rim)
    rho = np.sqrt(((xx - (ex0 + ex1) / 2) / ((ex1 - ex0) / 2)) ** 2 + ((yy - cy) / b) ** 2)
    text_px = diff_mask(img, x, y, x + w, y + h, thr=45, grow=1)
    prof = []
    for k_ in range(12):
        sel = comp & ~text_px & (rho >= k_ / 12) & (rho < (k_ + 1) / 12)
        prof.append(np.median(img[sel], 0) if sel.sum() > 3 else None)
    last = plate
    for k_ in range(12):
        if prof[k_] is None: prof[k_] = last
        last = prof[k_]
    # text
    cols = rgb_of(r['fill'])
    sh_col = r['sh_col']; soff = r['sh_off']
    size = 24
    a, base, _ = render(text, F_HEAD, size)
    while a.shape[0] > h + 2 and size > 14:
        size -= 1; a, base, _ = render(text, F_HEAD, size)
    tw = a.shape[1]
    margin = max(8, ((ex1 - ex0) - w) / 2)
    nw = max(ex1 - ex0, tw + 2 * margin)
    nx0 = ex0; nx1 = ex0 + nw
    ncx = (nx0 + nx1) / 2
    gy, gx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    t = np.clip((gx / max(1, a.shape[1]) * 0.5 + gy / max(1, a.shape[0]) * 0.5), 0, 1)
    # alternate light/dark per letter like the original's 2-tone glyphs
    colmap = cols[0] * (1 - t[..., None]) + cols[1] * t[..., None]
    tx = int(round(ncx - tw / 2)); ty = int(round(y + (h - a.shape[0]) / 2))
    def job():
        im = pg.img
        rr = np.sqrt(((xx - ncx) / (nw / 2)) ** 2 + ((yy - cy) / b) ** 2)
        for k_ in range(12):
            sel = (rr >= k_ / 12) & (rr < (k_ + 1) / 12)
            im[sel] = prof[k_]
        edge = (rr >= 0.97) & (rr < 1.06)
        wgt = np.clip((1.06 - rr) / 0.09, 0, 1)
        im[edge] = (im[edge] * (1 - wgt[edge, None]) + prof[11] * wgt[edge, None])
        over(im, tx + soff[0], ty + soff[1], a, sh_col)
        over(im, tx, ty, a, colmap)
    jobs.append(job)
    return f'{r["id"]}: size {size} plate {ex1 - ex0}->{nw:.0f}'

def grey_plate(pg, r, text, jobs):
    """Grey bevelled label plate: repaint the whole plate."""
    img = pg.img; shp = img.shape[:2]
    x, y, w, h = r['x'], r['y'], r['w'], r['h']
    mk = rect_mask(shp, x - 1, y - 1, w + 5, h + 5)
    pg.img = inpaint(img, mk, iters=150)
    size = 13
    a, base, _ = render(text, F_BODY, size)
    pw_ = max(w, a.shape[1] + 12)
    tx = x + 6; ty = int(round(y + (h - 1 - a.shape[0]) / 2))
    def job():
        im = pg.img
        im[y + 2:y + h + 2, x + 2:x + pw_ + 2] = (32, 24, 40)
        im[y:y + h, x:x + pw_] = (152, 152, 144)
        im[y, x:x + pw_] = (160, 160, 160)
        im[y + h - 2, x + 1:x + pw_] = (184, 176, 176)
        im[y + h - 1, x + 1:x + pw_] = (112, 104, 104)
        im[y:y + h, x] = (144, 144, 144)
        im[y:y + h - 1, x + pw_ - 1] = (128, 128, 128)
        over(im, tx + 1, ty + 1, a, np.array([16., 16, 16]))
        over(im, tx, ty, a, np.array([248., 248, 248]))
    jobs.append(job)
    return f'{r["id"]}: plate {w}->{pw_}'

def back_hint(pg, r, text, jobs):
    img = pg.img
    mk = ink_mask(img, r, tol=80, extra=1)
    pg.img = inpaint(img, mk, iters=100)
    fill = rgb_of(r['fill'])[0]
    sh = r['sh_col'] if r['sh_col'] is not None else None
    size = 13.5
    a, base, _ = render(text, F_BODY, size)
    while a.shape[1] > r['w'] + 4 and size > 10:
        size -= 0.5; a, base, _ = render(text, F_BODY, size)
    tx = r['x'] + r['w'] - a.shape[1]; ty = int(round(r['y'] + (r['h'] - a.shape[0]) / 2))
    def job():
        if sh is not None: over(pg.img, tx + 1, ty + 1, a, sh)
        over(pg.img, tx, ty, a, fill)
    jobs.append(job)
    return f'{r["id"]}: size {size} w {a.shape[1]} (rect {r["w"]})'

def paint_rules(pg, rows, tl):
    """All rows of one rules page: erase first, then draw."""
    jobs = []; log = []
    shp = pg.img.shape[:2]
    # 1. body / label text and botan captions: one combined mask
    mk = np.zeros(shp, bool); prot = np.zeros(shp, bool)
    for r in rows:
        if r['kind'] in ('body', 'label') and 'grey plate' not in r['notes']:
            mk |= erase_body(pg, r)
            prot |= protect_mask(shp, r, 1)
    mk &= ~protect_mask_all(shp, rows, 0)
    new = inpaint(pg.img, mk, ~prot, iters=200)
    pg.img = grain(new, mk, pg.img, 0.9, ignore=prot)
    for r in rows:
        t = tl.get(r['id'])
        if t is None: continue
        if r['kind'] in ('body', 'label') and 'grey plate' not in r['notes']:
            log.append(body(pg, r, t, jobs))
        elif r['kind'] == 'label':
            log.append(grey_plate(pg, r, t, jobs))
        elif r['kind'] == 'heading' and 'rainbow' in r['fill']:
            log.append(header_band(pg, r, t, jobs))
        elif r['kind'] == 'heading':
            log.append(section_heading(pg, r, t, jobs))
        elif r['kind'] == 'button_hint':
            log.append(back_hint(pg, r, t, jobs))
    for j in jobs: j()
    return log

# ---------------------------------------------------------------- titles

def poly_mask(shape, pts):
    im = Image.new('L', (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=255)
    return np.asarray(im) > 127

_plate = {}
def green_plate():
    """The t1 banner plate as (rgb, mask) in t1 coordinates. Its colour is
    constant along the 0.4 px/row slant, so each slanted column takes its
    commonest colour over the main rows (text covers only part of each)."""
    if _plate: return _plate['v']
    src = open('private/unpacked/MINITTL1.BIZ.lz', 'rb').read()
    img = Page(src, TITLE).img
    s = 0.4; top, bot = 56, 77
    rgb = np.zeros((PH, PW, 3)); m = np.zeros((PH, PW), bool)
    for x in range(54, 436):
        cols = []
        for y in range(68, bot + 1):
            xx = int(round(x + (bot - y) * s))
            cols.append(img[y, xx])
        cols = np.array(cols).astype(int)
        if x < 430:                     # text outline pixels are near black
            keep = cols.sum(1) >= 100
            if keep.any(): cols = cols[keep]
        key = cols[:, 0] * 65536 + cols[:, 1] * 256 + cols[:, 2]
        v, c = np.unique(key, return_counts=True)
        best = v[c.argmax()]
        col = np.array([best >> 16, best >> 8 & 255, best & 255], float)
        for y in range(top, bot + 1):
            xx = int(round(x + (bot - y) * s))
            rgb[y, xx] = col; m[y, xx] = True
    # left/right ends: fill the slant gaps from the neighbour
    m2 = ndimage.binary_closing(m, structure=np.ones((1, 3)))
    gap = m2 & ~m
    rgb[gap] = rgb[np.roll(gap, 1, 1)] if False else ndimage.grey_dilation(rgb, size=(1, 3, 1))[gap]
    m = m2
    # bottom strip rows 78-80 (dark lower edge) are text-free: copy them
    for y in range(bot + 1, bot + 4):
        xs = np.arange(54, 434)
        rgb[y, xs] = img[y, xs]; m[y, xs] = True
    _plate['v'] = (rgb, m)
    return _plate['v']

def lens(shape, x0, x1, ymid, thick, keys):
    """Spindle-shaped plate (t4/t8): body colour by x from keys [(x, rgb)],
    lighter top half, 2 px dark red rim at the bottom. Returns (rgb, alpha)."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    u = np.clip((xx - x0) / (x1 - x0), 0, 1)
    half = thick / 2 * np.sin(np.pi * u) ** 0.45
    d = yy - ymid                                    # <0 above the centre line
    a = np.clip(half + 0.5 - np.abs(d), 0, 1)
    a[(xx < x0) | (xx > x1)] = 0
    kx = np.array([k[0] for k in keys], float); kc = np.array([k[1] for k in keys], float)
    body = np.stack([np.interp(xx, kx, kc[:, c]) for c in range(3)], -1)
    light = np.clip(-d / np.maximum(half, 1), 0, 1)[..., None]
    col = body * (1 - 0.45 * light) + np.array([248., 232, 176]) * 0.45 * light
    rim = (d > half - 2.2) & (half > 1.5)
    col[rim] = (128, 16, 0)
    return col, a

def shift_fill(img, mk, shifts, todo):
    """Fill the todo pixels by copying from the first pixel outside mk at
    one of the given (dy, dx) shifts (periodic or directional texture)."""
    out = img.copy()
    todo = todo.copy()
    H, W = mk.shape
    for dy, dx in shifts:
        if not todo.any(): break
        ys, xs = np.where(todo)
        sy, sx_ = ys + dy, xs + dx
        ok = (sy >= 0) & (sy < H) & (sx_ >= 0) & (sx_ < W)
        ys, xs, sy, sx_ = ys[ok], xs[ok], sy[ok], sx_[ok]
        good = ~mk[sy, sx_]
        out[ys[good], xs[good]] = img[sy[good], sx_[good]]
        todo[ys[good], xs[good]] = False
    return out, todo

def title_alpha(text, path, size, shear, box_w, box_h, caps=False, track=0.0, sx_min=0.8):
    if caps: text = text.upper()
    while True:
        a, base, lab = render(text, path, size, shear=shear, per_letter=True, track=track)
        sx = min(1.0, box_w / a.shape[1])
        if (sx >= sx_min and a.shape[0] <= box_h) or size < 14: break
        size -= 1
    if sx < 1.0:
        a, base, lab = render(text, path, size, shear=shear, sx=sx, per_letter=True, track=track)
    return a, base, lab, size, sx

def title_lines(text, split, path, sizes, shear, box_w, box_h, track=0.0, gap=-2, sx_min=0.85):
    """Two-line title: text split after `split`; line heights in the ratio
    of sizes. Returns combined alpha and the line-1/line-2 boundary row."""
    if not split or split not in text:
        a, base, lab, size, sx = title_alpha(text, path, sizes[-1], shear, box_w, box_h, track=track,
                                             sx_min=sx_min)
        return a, [size], sx
    k = text.index(split) + len(split)
    parts = [text[:k].strip(), text[k:].strip()]
    tot = sum(sizes)
    out = []
    for t, sz in zip(parts, sizes):
        hh = box_h * sz / tot
        a, base, lab, size, sx = title_alpha(t, path, sz, shear, box_w, hh + 6, track=track, sx_min=sx_min)
        out.append((a, size, sx))
    W = max(a.shape[1] for a, _, _ in out)
    H = sum(a.shape[0] for a, _, _ in out) + gap
    A = np.zeros((H, W))
    y = 0
    for i, (a, _, _) in enumerate(out):
        x = (W - a.shape[1]) // 2 if i else int((W - a.shape[1]) * 0.15)
        A[y:y + a.shape[0], x:x + a.shape[1]] = np.maximum(A[y:y + a.shape[0], x:x + a.shape[1]], a)
        y += a.shape[0] + gap
    return A, [o[1] for o in out], min(o[2] for o in out)

def vgrad(h, w, stops):
    """Vertical gradient with stops [(t, rgb)]."""
    t = np.linspace(0, 1, h)
    ts = np.array([s[0] for s in stops]); cs = np.array([s[1] for s in stops], float)
    col = np.stack([np.interp(t, ts, cs[:, c]) for c in range(3)], -1)
    return np.broadcast_to(col[:, None, :], (h, w, 3)).copy()

def rough(a, amount=0.35, seed=0):
    """Dry-brush edge: perturb the alpha edge with smooth noise."""
    rng = np.random.default_rng(seed)
    n = ndimage.gaussian_filter(rng.standard_normal(a.shape), 0.8)
    n /= max(1e-6, n.std())
    b = np.clip(a + amount * n * (a > 0.05) * (a < 0.98 + 0.02), 0, 1)
    b = np.clip((b - 0.15) / 0.7, 0, 1)
    return b

def lattice(v1, v2, n=4):
    """Shifts k1*v1 + k2*v2, nearest first."""
    out = []
    for k1 in range(-n, n + 1):
        for k2 in range(-n, n + 1):
            if k1 or k2:
                dy = k1 * v1[0] + k2 * v2[0]; dx = k1 * v1[1] + k2 * v2[1]
                out.append((dy * dy + dx * dx, (dy, dx)))
    return [d for _, d in sorted(out)]

def along(v, n=8):
    """Shifts along one direction, both ways, nearest first."""
    out = []
    for k in range(1, n + 1):
        dy, dx = int(round(k * v[0])), int(round(k * v[1]))
        out += [(dy, dx), (-dy, -dx)]
    return out

PRED = {
    'blue': lambda c: (c[..., 2] > 140) & (c[..., 2] > c[..., 0] + 60) & (c[..., 2] > c[..., 1] + 50),
    'ghost': lambda c: (c[..., 0] > 215) & (c[..., 1] > 110) & (c[..., 2] < 130) | (
        (c[..., 0] > 215) & (c[..., 1] > 90) & (c[..., 1] < 200) & (c[..., 2] > 70) & (c[..., 2] < 160)),
    'darkred': lambda c: (c[..., 0] > 50) & (c[..., 0] < 235) & (c[..., 1] < 60) & (c[..., 2] < 24)
        & (c[..., 0] > c[..., 1] * 2.2),
    'light': lambda c: c.sum(-1) > 430,
    'echo': lambda c: (c[..., 1] > 140) & (c[..., 0] < 150) & (c[..., 2] > 60),
    'yellow': lambda c: (c[..., 0] > 190) & (c[..., 1] > 120) & (c[..., 2] < 80),
    'orangeghost': lambda c: (c[..., 0] > 220) & (c[..., 2] < 20) & (c[..., 1] > 50) & (c[..., 1] < 185),
    'redghost': lambda c: (c[..., 0] > 180) & (c[..., 1] < 120) & (c[..., 2] < 60),
}

# per title: erase rect (x0, y0, x1, y1) for the background-difference mask,
# protect polygons (kept except for colour ops), colour ops (polygon,
# predicate) applied anywhere, fill regions (polygon or None, shifts),
# text box, style.
T7_POLE = [(418, 0), (462, 0), (414, 106), (376, 106)]
T8_POLE = [(300, 0), (352, 0), (278, 84), (222, 84)]
T8_HAIR = [(96, 64), (112, 50), (150, 45), (178, 52), (180, 84), (96, 84)]
T4_HAIR = [(76, 60), (90, 52), (146, 50), (164, 58), (164, 78), (76, 78)]
T3_HEAD1 = [(86, 72), (100, 65), (126, 63), (142, 70), (170, 77), (172, 100), (86, 100)]
T3_HEAD2 = [(326, 80), (340, 67), (372, 65), (394, 78), (396, 100), (326, 100)]
T2_VISOR = [(80, 66), (90, 58), (97, 55), (130, 55), (142, 58), (150, 64), (178, 84), (80, 84)]
T7_RIGHT = [(462, 0), (512, 0), (512, 110), (414, 110)]
T7_HAIR = [(96, 0), (212, 0), (212, 106), (96, 106)]
TCFG = {
    1: dict(style='silver', erase=(58, 22, 448, 78), plate=(0, 0), box=(70, 26, 432, 76)),
    5: dict(style='silver', erase=(68, 10, 456, 66), plate=(9, -12), box=(80, 14, 400, 64)),
    2: dict(style='yellow', erase=(88, 8, 424, 84), box=(96, 4, 420, 82), split="Roberto's",
            protect=[T2_VISOR], colour=[(T2_VISOR, 'yellow'), (T2_VISOR, 'darkred'), (T2_VISOR, 'echo'),
                                         (T2_VISOR, 'light')]),
    6: dict(style='yellow', erase=(80, 14, 414, 86), box=(86, 8, 372, 86), split="Roberto's"),
    3: dict(style='brush', erase=(36, 16, 486, 98), box=(40, 22, 478, 84),
            protect=[T3_HEAD1, T3_HEAD2],
            colour=[(T3_HEAD1, 'darkred'), (T3_HEAD2, 'darkred')]),
    7: dict(style='brush', erase=None, box=(40, 24, 412, 82), ghost=(240, 112, 8),
            band=(26, 30, 432, 82, (248, 236, 204), 0.92),
            colour=[([(30, 22), (490, 22), (490, 96), (30, 96)], 'darkred'),
                    ([(30, 14), (110, 14), (110, 100), (30, 100)], 'redghost'),
                    ([(30, 14), (420, 14), (420, 100), (30, 100)], 'orangeghost'),
                    ([(400, 14), (490, 14), (490, 100), (400, 100)], 'ghost')],
            fill=[(T7_POLE, along((1, -0.442), 80)), (T7_RIGHT, []),
                  (T7_HAIR, []), (None, lattice((9, 7), (9, -7), 4))]),
    4: dict(style='script', erase=(40, 10, 482, 70), box=(56, 12, 470, 66),
            lens=(46, 466, 42.5, 18), protect=[T4_HAIR],
            colour=[(T4_HAIR, 'blue')]),
    8: dict(style='script', erase=(28, 8, 502, 72), box=(40, 12, 414, 64),
            lens=(34, 470, 43, 18), protect=[T8_POLE, T8_HAIR],
            colour=[(T8_POLE, 'blue'), (T8_POLE, 'ghost'), (T8_HAIR, 'blue'), (T8_HAIR, 'ghost')],
            fill=[(T8_POLE, along((1, -1.133), 80)), (None, along((-2, 3), 20))], lens_under=[T8_POLE],
            bubble_shift=True),
}

def paint_title(pg, n, rows, tl):
    """Title picture n: erase title, stamp and bubble text, then redraw."""
    cfg = TCFG[n]; img = pg.img; shp = img.shape[:2]; log = []
    br = rows[f't{n}_bubble']; sr = rows.get(f't{n}_stamp')
    prot = np.zeros(shp, bool)
    for poly in cfg.get('protect', []): prot |= poly_mask(shp, poly)
    mk = np.zeros(shp, bool)
    smk = stamp_mask(img, sr) if sr is not None else np.zeros(shp, bool)
    plate = cfg.get('plate')
    pm = np.zeros(shp, bool)
    if plate is not None:
        prgb, pm = green_plate()
        dx, dy = plate
        prgb = np.roll(np.roll(prgb, dy, 0), dx, 1); pm = np.roll(np.roll(pm, dy, 0), dx, 1)
    if cfg.get('erase'):
        x0, y0, x1, y1 = cfg['erase']
        mk |= diff_mask(img, x0, y0, x1, y1, thr=cfg.get('thr', 32), grow=2, unknown=smk | pm) & ~prot
    for poly, pred in cfg.get('colour', []):
        m = PRED[pred](img) & poly_mask(shp, poly)
        mk |= ndimage.binary_dilation(m, iterations=2) & poly_mask(shp, poly)
    mk |= smk
    mk &= ~pm
    new = img.copy(); rest = mk.copy()
    claimed = np.zeros(shp, bool)
    for poly, shifts in cfg.get('fill', []):
        todo = rest & ~claimed if poly is None else rest & poly_mask(shp, poly) & ~claimed
        if poly is not None: claimed |= poly_mask(shp, poly)
        new, left = shift_fill(img, mk, shifts, todo)
        rest &= ~(todo & ~left)
        img = new
    new = img
    if (rest & prot).any():
        # strokes over protected art: fill from that art only
        img = inpaint(img, rest & prot, prot, iters=300)
        rest &= ~prot
    new = img
    if rest.any():
        new = inpaint(img, rest, (~prot | rest) & ~pm, iters=300)
        new = grain(new, rest, pg.img, 1.0, ignore=pm | smk)
    if plate is not None:
        new[pm] = prgb[pm]
    if 'lens' in cfg:
        lx0, lx1, ly, lt = cfg['lens']
        L = lx1 - lx0
        keys = [(lx0, (232, 112, 8)), (lx0 + 0.15 * L, (240, 168, 88)), (lx0 + 0.35 * L, (248, 216, 136)),
                (lx0 + 0.55 * L, (240, 168, 96)), (lx0 + 0.78 * L, (192, 88, 24)),
                (lx1 - 14, (248, 224, 160)), (lx1, (248, 232, 168))]
        lc, la = lens(shp, lx0, lx1, ly, lt, keys)
        for poly in cfg.get('lens_under', []): la = la * ~poly_mask(shp, poly)
        new = new * (1 - la[..., None]) + lc * la[..., None]
    if 'band' in cfg:
        # a brush-stroke paper band behind the title (hides the busy fill)
        bx0, by0, bx1, by1, bcol, balpha = cfg['band']
        rng = np.random.default_rng(n)
        yy, xx = np.mgrid[0:shp[0], 0:shp[1]].astype(float)
        wob_t = ndimage.gaussian_filter1d(rng.standard_normal(shp[1]), 6) * 6
        wob_b = ndimage.gaussian_filter1d(rng.standard_normal(shp[1]), 6) * 6
        wob_l = ndimage.gaussian_filter1d(rng.standard_normal(shp[0]), 2) * 8
        wob_r = ndimage.gaussian_filter1d(rng.standard_normal(shp[0]), 2) * 8
        inside_ = ((yy > by0 + wob_t[None, :]) & (yy < by1 + wob_b[None, :]) &
                   (xx > bx0 + wob_l[:, None]) & (xx < bx1 + wob_r[:, None]))
        ba = ndimage.gaussian_filter(inside_.astype(float), 0.8)
        streak = 1 + 0.06 * ndimage.gaussian_filter1d(rng.standard_normal(shp[0]), 1)[:, None]
        col = np.clip(np.array(bcol, float)[None, None, :] * streak[..., None], 0, 255)
        sh_ = np.roll(np.roll(ba, 3, 0), 3, 1)
        new = new * (1 - 0.45 * sh_[..., None]) + np.array([64., 16, 0]) * 0.45 * sh_[..., None]
        new = new * (1 - balpha * ba[..., None]) + col * balpha * ba[..., None]
    pg.img = new
    if os.environ.get('FESTPICS_ERASE_ONLY'): return log
    log.append(draw_title(pg, cfg, tl[f't{n}_title'], n))
    log.append(bubble(pg, br, tl[f't{n}_bubble'], cfg))
    if sr is not None: log.append(stamp(pg, sr, tl[f't{n}_stamp']))
    return log

def draw_title(pg, cfg, text, n):
    img = pg.img
    bx0, by0, bx1, by1 = cfg['box']; bw, bh = bx1 - bx0, by1 - by0
    st = cfg['style']
    if st == 'silver':
        a, base, lab, size, sx = title_alpha(text, F_IMPACT, 40, 0.22, bw - 6, bh - 4, caps=True, track=1.0, sx_min=0.88)
        X = int(round(bx0 + (bw - 4 - a.shape[1]) / 2)); Y = by1 - 4 - a.shape[0]
        o = dilate(a, 1)
        for k in (3, 2, 1):
            over(img, X - 1 + k, Y - 1 + k, o, np.array([0., 32, 0]))
        over(img, X - 1, Y - 1, o, np.array([224., 240, 236]))
        g = vgrad(a.shape[0], a.shape[1], [(0, (248, 248, 248)), (0.45, (184, 208, 220)),
                                           (0.7, (152, 184, 200)), (1, (208, 228, 236))])
        over(img, X, Y, a, g)
    elif st == 'yellow':
        a, size, sx = title_lines(text, cfg.get('split'), F_HEAD, (26, 40), 0.0, bw - 12, bh - 8,
                                  track=0.5)
        X = int(round(bx0 + (bw - 8 - a.shape[1]) / 2)); Y = int(round(by0 + 6 + (bh - 8 - a.shape[0]) / 2))
        o = dilate(a, 1)
        # glitch echo: cyan/green/blue copy up-right, rows jittered
        e = np.zeros((o.shape[0] + 4, o.shape[1] + 16))
        rng = np.random.default_rng(n)
        jit = 0
        ec = np.zeros(e.shape + (3,))
        palette = [(32, 200, 232), (64, 216, 120), (32, 160, 216), (32, 112, 200), (120, 224, 200)]
        for yy in range(o.shape[0]):
            if yy % 2 == 0: jit = int(rng.integers(0, 7))
            e[yy, jit:jit + o.shape[1]] = np.maximum(e[yy, jit:jit + o.shape[1]], o[yy])
            ec[yy] = palette[(yy // 3) % len(palette)]
        # streak the echo to the right
        for k in range(1, 5):
            e[:, k:] = np.maximum(e[:, k:], e[:, :-k] * (1 - k * 0.2))
        over(img, X + 6, Y - 7, e, ec, 0.85)
        over(img, X + 2, Y + 2, o, np.array([48., 0, 0]))
        over(img, X, Y, o, np.array([72., 16, 0]))
        g = vgrad(a.shape[0], a.shape[1], [(0, (248, 248, 120)), (0.3, (248, 240, 24)),
                                           (0.75, (248, 200, 8)), (1, (232, 152, 8))])
        over(img, X + 1, Y + 1, a, g)
    elif st == 'brush':
        a, base, lab, size, sx = title_alpha(text, F_SERIF, 46, 0.0, bw - 16, bh - 6, track=0.0)
        a = np.pad(a, ((0, 1), (0, 1)))
        a[1:, 1:] = np.maximum(a[1:, 1:], a[:-1, :-1] * 0.9)
        a[:, 1:] = np.maximum(a[:, 1:], a[:, :-1])
        a = rough(a, 0.5, n)
        X = int(round(bx0 + (bw - 12 - a.shape[1]) / 2)); Y = int(round(by0 + (bh - a.shape[0]) / 2))
        ghost = ndimage.gaussian_filter(dilate(a, 2), 1.4)
        gc = np.array(cfg.get('ghost', (240, 112, 8)), float)
        over(img, X + 9, Y + 4, ghost, gc, 0.9)
        o = dilate(a, 1)
        over(img, X + 1, Y + 1, o, np.array([64., 8, 0]), 0.6)
        g = vgrad(a.shape[0], a.shape[1], [(0, (152, 40, 8)), (0.6, (120, 32, 0)), (1, (88, 24, 0))])
        over(img, X, Y, a, g)
    elif st == 'script':
        a, base, lab, size, sx = title_alpha(text, F_SCRIPT, 40, 0.0, bw - 12, bh - 2, track=0.0)
        X = int(round(bx0 + (bw - 8 - a.shape[1]) / 2)); Y = int(round(by0 + (bh - a.shape[0]) / 2))
        # pink / yellow ghost strokes behind alternate letters
        for i in range(lab.max() + 1 if lab is not None else 0):
            li = (lab == i) * a
            if not li.any(): continue
            gg = dilate(li, 2)
            if i % 3 == 1:
                over(img, X - 6, Y + 2, gg, np.array([248., 152, 112]), 0.9)
            elif i % 3 == 2:
                over(img, X + 5, Y + 3, gg, np.array([248., 208, 64]), 0.9)
        over(img, X + 1, Y + 1, a, np.array([0., 16, 120]), 0.5)
        over(img, X, Y, a, np.array([0., 32, 248]))
    return f't{n}_title: {st} size {size} sx {sx:.2f} w {a.shape[1]}'

def bubble(pg, r, text, cfg):
    """Flat speech bubble: fill the text with the bubble colour, then draw
    the EN centred in the same text colour and shadow."""
    img = pg.img; shp = img.shape[:2]
    x, y, w, h = r['x'], r['y'], r['w'], r['h']
    R = rect_mask(shp, x, y, w, h, 3)
    bcol = mode_colour(img, rect_mask(shp, x, y, w, h))
    near = np.sqrt(((img - bcol) ** 2).sum(-1)) < 20
    lab, _ = ndimage.label(near)
    k = np.bincount(lab[R].ravel()); k[0] = 0
    comp = lab == k.argmax()
    ys_, xs_ = np.where(comp)
    diamond = r['lines'] == 3            # the practice bubbles are rhombi
    if diamond:
        t_ = (xs_[ys_ == ys_.min()].mean(), ys_.min()); b_ = (xs_[ys_ == ys_.max()].mean(), ys_.max())
        l_ = (xs_.min(), ys_[xs_ == xs_.min()].mean())
        # text can cut off the right corner: complete the parallelogram
        r_ = (t_[0] + b_[0] - l_[0], t_[1] + b_[1] - l_[1])
        inside = poly_mask(shp, [t_, r_, b_, l_])
    else:
        yy_, xx_ = np.mgrid[-5:6, -5:6]
        inside = ndimage.binary_fill_holes(
            ndimage.binary_closing(comp, structure=yy_ ** 2 + xx_ ** 2 <= 25, border_value=0) | comp)
    inner = ndimage.binary_erosion(inside, iterations=1)
    fill_c = rgb_of(r['fill'])[0]
    tc = np.sqrt(((img - fill_c) ** 2).sum(-1)) < 120
    if r['sh_col'] is not None:
        tc |= np.sqrt(((img - r['sh_col']) ** 2).sum(-1)) < 60
    mk = R & ((inner & ~near) | (inside & ndimage.binary_dilation(tc, iterations=1)))
    img[mk] = bcol
    # JP text that runs past the bubble edge: text and shadow colours
    # outside the bubble, filled like the title background
    ring = ~inside
    ts = np.sqrt(((img - fill_c) ** 2).sum(-1)) < 70
    if r['sh_col'] is not None:
        ts |= np.sqrt(((img - r['sh_col']) ** 2).sum(-1)) < 40
    cand = ts & rect_mask(shp, x, y, w, h, 3) & ring
    lab2, nl = ndimage.label(cand, structure=np.ones((3, 3)))
    touch = np.unique(lab2[ndimage.binary_dilation(inside, iterations=2) & cand])
    if r['sh_col'] is not None:
        # a text fragment carries its dark drop shadow
        shp_ = np.sqrt(((img - r["sh_col"]) ** 2).sum(-1)) < 55
        shd = np.unique(lab2[ndimage.binary_dilation(shp_, iterations=3) & cand])
        touch = np.intersect1d(touch, shd)
    cand = np.isin(lab2, touch[touch > 0])
    outm = ndimage.binary_dilation(cand, iterations=3) & ring
    outm &= rect_mask(shp, x, y, w, h, 6)
    if outm.any():
        shifts = [sh_ for poly, sh_ in cfg.get('fill', []) if poly is None and cfg.get('bubble_shift')]
        rest = outm
        if shifts:
            img, rest = shift_fill(img, outm | ndimage.binary_dilation(inside, iterations=9),
                                   shifts[0], outm)
        if rest.any():
            new = inpaint(img, rest, ~inside, iters=200)
            img = grain(new, rest, img, 1.0, ignore=inside)
        pg.img = img
    # lines are centred on the bubble interior at their own row
    lines = text.split('|')
    fill = rgb_of(r['fill'])[0]
    sh = r['sh_col']; so = r['sh_off']
    n = len(lines)
    ys = np.where(inside.any(1))[0]
    cyb = (ys.min() + ys.max()) / 2
    def extents(pitch):
        top = cyb - pitch * n / 2
        out = []
        for i in range(n):
            ly = int(round(top + i * pitch + pitch / 2))
            band = inner[ly - 3:ly + 4].all(0)
            xs_ = np.where(band)[0]
            if not len(xs_): return None, top
            out.append((xs_.min(), xs_.max()))
        return out, top
    best = None
    for size in np.arange(18, 10.9, -0.5):
        pitch = round(size * 1.35)
        ext, top = extents(pitch)
        if ext is None: continue
        for sx in (1.0, 0.94, 0.9):
            if all(text_width(t, F_BODY, size, sx) <= (b - a) - 8 for t, (a, b) in zip(lines, ext)):
                best = (size, sx, pitch, ext, top); break
        if best: break
    if best is None:
        size, sx = 11, 0.9; pitch = 16; ext, top = extents(pitch)
    else:
        size, sx, pitch, ext, top = best
    for i, t in enumerate(lines):
        a_, b_ = ext[i]
        draw_lines(pg.img, [t], int(a_), top + i * pitch, F_BODY, size, sx, pitch, fill, sh, so,
                   align='centre', width=b_ - a_)
    return f'{r["id"]}: size {size} sx {sx:.2f}'

def stamp_mask(img, r):
    shp = img.shape[:2]
    R = rect_mask(shp, r['x'], r['y'], r['w'], r['h'], 12)
    rr, gg, bb = img[..., 0], img[..., 1], img[..., 2]
    red = (rr > 40) & (rr > gg * 2.0 + 16) & (rr > bb * 2.0 + 16)
    m = ndimage.binary_closing(red & R, iterations=2)
    m = ndimage.binary_fill_holes(m)
    return ndimage.binary_dilation(m, iterations=3) & R

def stamp(pg, r, text):
    """Red rubber stamp: framed heavy condensed text, rotated ~13 deg
    (right side up) like the original 練習用."""
    ss = 4
    f = font(F_IMPACT, 24 * ss)
    t = Image.new('L', (int(f.getlength(text)) + 16 * ss, 40 * ss), 0)
    ImageDraw.Draw(t).text((8 * ss, 4 * ss), text, font=f, fill=255)
    t = t.crop(t.getbbox())
    tw = min(t.width, (r['w'] - 14) * ss)
    t = t.resize((tw, t.height), Image.LANCZOS)
    W, H = tw + 14 * ss, t.height + 14 * ss
    im = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(im)
    d.rectangle([1 * ss, 1 * ss, W - 1 - 1 * ss, H - 1 - 1 * ss], outline=255, width=int(2.5 * ss))
    d.rectangle([4.5 * ss, 4.5 * ss, W - 1 - 4.5 * ss, H - 1 - 4.5 * ss], outline=255, width=int(1 * ss))
    im.paste(255, ((W - t.width) // 2, (H - t.height) // 2), t)
    im = im.rotate(13, resample=Image.BICUBIC, expand=True)
    a = np.asarray(im.resize((im.width // ss, im.height // ss), Image.BOX), float) / 255
    a = np.clip(a * 1.2, 0, 1)
    cx, cy = r['x'] + r['w'] / 2, r['y'] + r['h'] / 2
    X, Y = int(round(cx - a.shape[1] / 2)), int(round(cy - a.shape[0] / 2))
    over(pg.img, X + 1, Y + 1, dilate(a, 1)[1:-1, 1:-1], r['sh_col'])
    over(pg.img, X, Y, a, np.array([232., 0, 0]))
    return f'{r["id"]}: {a.shape[1]}x{a.shape[0]}'

# ---------------------------------------------------------------- CEFT5 HUD

CEFT_TIM, CEFT_PIX, CEFT_W, CEFT_H = 0x10800, 0x11020, 768, 256
# id: (x, y, w, h, clut, kind); kind 'plate' = white text on the orange bar
HUD = {
    'hud_konjo': (0, 56, 88, 16, 4, 'plate'),
    'hud_ato_kyu': (0, 72, 128, 16, 4, 'plate'),
    'hud_genzai': (720, 128, 48, 16, 2, 'small'),
    'hud_saiko': (200, 240, 56, 16, 2, 'small'),
    'hud_ten': (240, 0, 16, 16, 2, 'small'),
    'hud_geijutsu': (216, 56, 40, 16, 2, 'small'),
    'hud_frame': (136, 72, 40, 16, 2, 'small'),
}
# old text columns on the plates (left part, right part)
HUD_TEXT = {'hud_konjo': [(0, 62)], 'hud_ato_kyu': [(0, 56), (100, 128)]}

def hud_quant(rgb, cov, pal, allowed):
    """Nearest palette index (from allowed) for covered pixels, else 0."""
    out = np.zeros(cov.shape, np.uint8)
    P = pal[allowed]
    d = ((rgb[:, :, None, :] - P[None, None]) ** 2).sum(-1)
    q = np.array(allowed)[d.argmin(-1)]
    out[cov] = q[cov]
    return out

def hud_text(text, size, shear, path=F_BODY, ss=4):
    f = font(path, int(size * ss))
    im = Image.new('L', (int(f.getlength(text)) + 12 * ss, int(size * 2 * ss)), 0)
    ImageDraw.Draw(im).text((4 * ss, 2 * ss), text, font=f, fill=255)
    if shear:
        im = im.transform(im.size, Image.AFFINE, (1, shear, -shear * im.height * 0.7, 0, 1, 0),
                          resample=Image.BICUBIC)
    a = np.asarray(im.resize((im.width // ss, im.height // ss), Image.BOX), float) / 255
    rows_ = np.where(a.max(1) > 0.05)[0]; cols = np.where(a.max(0) > 0.05)[0]
    return a[rows_[0]:rows_[-1] + 1, cols[0]:cols[-1] + 1]

# end-of-round box (one picture shared by the Festival minigames, task A2):
# rect in the atlas, CLUT index, fill area, question and choice centres.
AGAIN = dict(x=656, y=168, w=112, h=72, clut=49, fill=(2, 14, 110, 70),
             q=('hud_again', 56, 31, 100, 19), yes=('hud_yes', 33, 55, 38),
             no=('hud_no', 76, 55, 38))

def again_draw(b, pal, a, x0, y0, top, bot, ramp):
    """Text a at (x0, y0): vertical gradient top->bot, 1 px outline and a
    +1,+1 shadow in index 1, quantized to [1] + ramp."""
    h, w = b.shape
    f = np.zeros((h, w)); f[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    o = ndimage.grey_dilation(f, footprint=np.ones((3, 3)))
    s = np.zeros((h, w)); s[1:, 1:] = o[:-1, :-1]
    dark = np.maximum(s, o)
    t = np.clip((np.arange(h)[:, None] - y0) / max(1, a.shape[0] - 1), 0, 1)[..., None]
    rgb = pal[1] * (1 - f[..., None]) + (pal[top] * (1 - t) + pal[bot] * t) * f[..., None]
    cov = (f > 0.3) | (dark > 0.4)
    q = hud_quant(rgb, cov, pal, [1] + ramp)
    b[cov] = q[cov]

def paint_again(px, d, tl, log):
    """Erase the JP text of the box to the fill stipple and draw the English
    question (cyan ramp 2-7) and choices (white ramp 8-12)."""
    A = AGAIN
    if not all(A[k][0] in tl for k in ('q', 'yes', 'no')): return None
    clut = struct.unpack_from('<1024H', d, CEFT_TIM + 0x14)
    pal = np.array([rgb15(v) for v in clut[A['clut'] * 16:A['clut'] * 16 + 16]])
    b = px[A['y']:A['y'] + A['h'], A['x']:A['x'] + A['w']]
    x0, y0, x1, y1 = A['fill']
    yy, xx = np.mgrid[0:A['h'], 0:A['w']]
    m = np.zeros(b.shape, bool); m[y0:y1, x0:x1] = True
    m &= ~np.isin(b, (13, 14))                   # keep the frame
    b[m] = np.where((xx + yy) % 2 == 1, 15, 0)[m]
    def txt(s, size, maxw, maxh):
        while True:
            a = hud_text(s, size, 0.0, F_BODY)
            if (a.shape[1] + 3 <= maxw and a.shape[0] + 3 <= maxh) or size <= 9: break
            size -= 1
        return np.clip((a - 0.15) / 0.6, 0, 1), size
    key, cx, cy, mw, mh = A['q']
    a, sz = txt(tl[key], 15, mw, mh)
    x, y = cx - (a.shape[1] + 1) // 2, cy - (a.shape[0] + 1) // 2
    again_draw(b, pal, a, x, y, 7, 5, list(range(2, 8)))
    log.append(f'{key}: "{tl[key]}" {a.shape[1]}x{a.shape[0]} size {sz} at {x},{y}')
    for k in ('yes', 'no'):
        key, cx, cy, mw = A[k]
        a, sz = txt(tl[key], 11, mw, 12)
        x, y = cx - (a.shape[1] + 1) // 2, cy - (a.shape[0] + 1) // 2
        again_draw(b, pal, a, x, y, 12, 11, list(range(8, 13)))
        log.append(f'{key}: "{tl[key]}" {a.shape[1]}x{a.shape[0]} size {sz} at {x},{y}')
    pal[0] = (255, 0, 255)
    return pal

def paint_hud(d, tl, png):
    clut = struct.unpack_from('<512H', d, CEFT_TIM + 0x14)
    raw = np.frombuffer(bytes(d[CEFT_PIX:CEFT_PIX + 384 * CEFT_H]), np.uint8).reshape(CEFT_H, 384)
    px = np.empty((CEFT_H, CEFT_W), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    before = px.copy(); log = []
    for rid, (x, y, w, h, c, kind) in HUD.items():
        if rid not in tl: continue
        pal = np.array([rgb15(v) for v in clut[c * 16:c * 16 + 16]])
        sub = px[y:y + h, x:x + w]
        parts = tl[rid].split('|')
        if kind == 'plate':
            # restore the bar under the old text from a clean column
            clean = sub[:, 80].copy()
            for (a0, a1) in HUD_TEXT[rid]:
                for r in range(h):
                    for xx in range(a0, a1):
                        left = sub[r, :].nonzero()[0]
                        lx = left.min() if len(left) else w
                        sub[r, xx] = clean[r] if (xx >= lx and r >= 8) else 0
            # text: white fill, 1 px dark outline, grey ramp antialias
            a = hud_text(parts[0], 14, 0.25, F_HEAD)
            if a.shape[0] > 13:
                a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
                    (max(1, int(a.shape[1] * 13 / a.shape[0])), 13), Image.LANCZOS), float) / 255
            ox, oy = 6, 0
            span = HUD_TEXT[rid][0][1] - ox
            if a.shape[1] + 2 > span: log.append(f'{rid}: "{parts[0]}" {a.shape[1]} px > {span}')
            o = np.zeros((h, w)); f_ = np.zeros((h, w))
            hh, ww = min(a.shape[0], h - oy - 1), min(a.shape[1], w - ox - 1)
            f_[oy + 1:oy + 1 + hh, ox + 1:ox + 1 + ww] = a[:hh, :ww]
            o = np.clip(ndimage.grey_dilation(f_, footprint=np.ones((3, 3))) * 1.6, 0, 1)
            base = pal[sub].astype(float)
            cov = (sub > 0) | (o > 0.3)
            rgb = base * (1 - o[..., None]) + pal[1] * o[..., None]
            rgb = rgb * (1 - f_[..., None]) + pal[15] * f_[..., None]
            q = hud_quant(rgb, cov, pal, list(range(1, 16)))
            # keep the bar pixels that the text does not touch
            keep = o < 0.3
            q[keep] = sub[keep]
            sub[:] = q
            log.append(f'{rid}: "{parts[0]}" w {a.shape[1]}')
        else:
            sub[:] = 0
            a = hud_text(parts[0], 11, 0.0, F_BODY)
            if a.shape[0] > 10:
                a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
                    (max(1, int(a.shape[1] * 10 / a.shape[0])), 10), Image.LANCZOS), float) / 255
            if a.shape[1] + 1 > w:
                log.append(f'{rid}: "{parts[0]}" {a.shape[1] + 1} px > {w}; squeezed')
                a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
                    (w - 1, a.shape[0]), Image.LANCZOS), float) / 255
            a = np.clip((a - 0.15) / 0.6, 0, 1)          # crisper strokes
            f_ = np.zeros((h, w)); f_[0:a.shape[0], 0:a.shape[1]] = a
            s_ = np.zeros((h, w)); s_[1:, 1:] = f_[:-1, :-1]
            s_ = np.maximum(s_, 0)
            cov = (f_ > 0.15) | (s_ > 0.3)
            rgb = np.zeros((h, w, 3)) + pal[2] * (s_[..., None] > 0.3)
            rgb = rgb * (1 - f_[..., None]) + pal[14] * f_[..., None]
            sub[:] = hud_quant(rgb, cov, pal, list(range(2, 15)))
            log.append(f'{rid}: "{parts[0]}" w {a.shape[1]} of {w}')
    apal = paint_again(px, d, tl, log)
    d[CEFT_PIX:CEFT_PIX + 384 * CEFT_H] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()
    if png:
        os.makedirs(PREV, exist_ok=True)
        rows_ = []
        for rid, (x, y, w, h, c, kind) in HUD.items():
            pal = np.array([rgb15(v) for v in clut[c * 16:c * 16 + 16]])
            pal[0] = (255, 0, 255)
            row = np.zeros((h, 2 * 128 + 8, 3)); row[:] = (40, 40, 40)
            row[:, :w] = pal[before[y:y + h, x:x + w]]
            row[:, 136:136 + w] = pal[px[y:y + h, x:x + w]]
            rows_.append(row); rows_.append(np.full((4, row.shape[1], 3), 40.0))
        sheet = np.concatenate(rows_).astype(np.uint8)
        Image.fromarray(sheet).resize((sheet.shape[1] * 4, sheet.shape[0] * 4), Image.NEAREST).save(
            f'{PREV}/hud.png')
        if apal is not None:
            A = AGAIN; sl = np.s_[A['y']:A['y'] + A['h'], A['x']:A['x'] + A['w']]
            sheet = np.zeros((A['h'], 2 * A['w'] + 8, 3)) + 40
            sheet[:, :A['w']] = apal[before[sl]]; sheet[:, A['w'] + 8:] = apal[px[sl]]
            Image.fromarray(sheet.astype(np.uint8)).resize(
                (sheet.shape[1] * 3, sheet.shape[0] * 3), Image.NEAREST).save(f'{PREV}/again.png')
    return log

# ---------------------------------------------------------------- build

def src_path(n):
    p = f'private/build/MINITTL{n}.BIZ.lz'
    return p if os.path.exists(p) else f'private/unpacked/MINITTL{n}.BIZ.lz'

def slot(name):
    s = os.path.getsize(f'private/disc/DAT2/{name}')
    return s + (-s % 2048)

def save_prev(name, before, after):
    os.makedirs(PREV, exist_ok=True)
    h, w = before.shape[:2]
    im = Image.new('RGB', (w * 2 + 8, h), (0, 0, 0))
    im.paste(Image.fromarray(before.astype(np.uint8)), (0, 0))
    im.paste(Image.fromarray(after.astype(np.uint8)), (w + 8, 0))
    im.save(f'{PREV}/{name}.png')

def build(png=False, only=None, dry=False):
    tl = load_tl(); rows = load_rows()
    problems = 0
    orig = {n: open(f'private/unpacked/MINITTL{n}.BIZ.lz', 'rb').read() for n in range(1, 9)}
    out = {n: bytearray(open(src_path(n), 'rb').read()) for n in range(1, 9)}
    # rules pages: paint in 1-4 (and 2's second page), copy into 5-8
    for n, off, tag in ((1, RULES, 'r1'), (2, RULES, 'r2a'), (2, RULES2, 'r2b'),
                        (3, RULES, 'r3'), (4, RULES, 'r4')):
        if only and tag not in only: continue
        pg = Page(orig[n], off)
        sel = [r for r in rows.values() if r['id'].startswith(tag + '_') and r['off'] == off]
        for line in paint_rules(pg, sel, tl): print(line)
        after = pg.store(out[n])
        blk = slice(off + 0x14, off + 0x220 + PW * PH)
        assert orig[n + 4][blk] == orig[n][blk]
        out[n + 4][blk] = out[n][blk]
        if png: save_prev(tag, pg.orig, after)
    for n in range(1, 9):
        if only and f't{n}' not in only: continue
        pg = Page(orig[n], TITLE)
        for line in paint_title(pg, n, rows, tl): print(line)
        after = pg.store(out[n])
        if png: save_prev(f't{n}', pg.orig, after)
    for n in range(1, 9):
        if dry: break
        c = lz.compress(bytes(out[n]))
        name = f'MINITTL{n}.BIZ'; s = slot(name)
        print(f'{name}: {len(c)} bytes, slot {s}')
        if len(c) > s: print(f'{name}: too large'); problems += 1; continue
        open(f'private/build/{name}.lz', 'wb').write(out[n])
        open(f'private/build/{name}', 'wb').write(c)
    if not only or 'hud' in only:
        src = 'private/unpacked/CEFT5.BIZ.lz'      # no other tool edits CEFT5
        d = bytearray(open(src, 'rb').read())
        for line in paint_hud(d, tl, png): print(line)
        if not dry:
            c = lz.compress(bytes(d)); s_ = slot('CEFT5.BIZ')
            print(f'CEFT5.BIZ: {len(c)} bytes, slot {s_}')
            if len(c) > s_: print('CEFT5.BIZ: too large'); problems += 1
            else:
                open('private/build/CEFT5.BIZ.lz', 'wb').write(d)
                open('private/build/CEFT5.BIZ', 'wb').write(c)
    print(f'festpics: problems {problems}')

if __name__ == '__main__':
    only = [a for a in sys.argv[1:] if not a.startswith('--')]
    build('--png' in sys.argv, only or None, '--dry' in sys.argv)
