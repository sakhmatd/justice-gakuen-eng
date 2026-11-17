# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Redraw the mode select screen graphics (DAT2/MENU.BIZ) in English.

Reads tl/TITLE.txt (ID: EN) and the rectangles in
data/boot_text.tsv. Writes private/build/MENU.BIZ (+ .lz) and edits the
description sprite lists in private/build/SLPS_012.40, so run it after hook.py.
Also redraws the Extras > Diary help panel (help_* in TITLE.txt) in both
copies of the MOJI2 font TIM and writes private/build/MOJI2.TIZ (+ .lz).
--png writes previews to private/re/boot/en/.

MENU.BIZ (decompressed):
- TIM1 at 0x1e800: CLUTs at VRAM (0,481) 512x2, pixels 4bpp 512x256 at
  0x1f020 (256 bytes per row). Header and mode labels.
- TIM4 at 0x52000: no CLUT, 4bpp 256x2132 at 0x52014 (128 bytes per row).
  One 256x164 block per mode; text at u0 v0 and a tail at u0 v80 that the
  game draws to the right of it. Palette CLUT (16,482).
Description sprite lists: 13 pointers at exe RAM 0x80083834; each list is
s16 count + 11 s16 per sprite (flag, tpage x, y, clut x, y, u, v, w, h,
screen x, y). The builder sets the text sprite to 255x80 and, where a tail
sprite exists, the tail to u0 v80 96x80, so every mode has 3 lines.
"""
import sys, os, re, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import lz

SRC = 'private/unpacked/MENU.BIZ.lz'
SLOT = 188 * 2048                      # disc allocation of MENU.BIZ
CLUT = 0x1e800 + 8 + 12                # TIM1 CLUT data, 512x2 entries
T1, T1W = 0x1f020, 512                 # TIM1 pixels
T4, T4W = 0x52014, 256                 # TIM4 pixels
FONT = '/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf'
FONTH = '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf'
EXE = 'private/build/SLPS_012.40'
LISTS = 0x80083834
def F(a): return a - 0x80010000 + 0x800

def load_tl(path='tl/TITLE.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        k, v = line.split(':', 1)
        out[k.strip()] = v.strip()
    return out

def rects():
    """id -> (x, y, w, h) in TIM1 texels, from boot_text.tsv."""
    r = {}
    for line in open('data/boot_text.tsv', encoding='utf-8'):
        m = re.match(r'(hdr\d|label\d+)\t.*?px\((\d+),(\d+)\) (\d+)x(\d+)', line)
        if m: r[m[1]] = tuple(int(m[i]) for i in range(2, 6))
    return r

def unpack(d, off, w, h):
    raw = np.frombuffer(bytes(d[off:off + w * h // 2]), np.uint8).reshape(h, w // 2)
    px = np.empty((h, w), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    return px

def pack(d, off, px):
    d[off:off + px.size // 2] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()

def lum(v): return (v & 31) * 2 + (v >> 5 & 31) * 3 + (v >> 10 & 31)

def palette(d, cx, cy):
    e = (cy - 481) * 512 + cx
    return struct.unpack_from('<16H', d, CLUT + 2 * e)

def mask(lines, w, h, size, font=FONT, ss=4, align='left', pitch=None):
    """Alpha (h, w) for the given lines; shrinks/squeezes to fit."""
    pitch = pitch or h // len(lines)
    while True:
        f = ImageFont.truetype(font, int(size * ss))
        boxes = [f.getbbox(t) for t in lines]
        asc = min(b[1] for b in boxes); desc = max(b[3] for b in boxes)
        tw = max(b[2] - b[0] for b in boxes)
        sx = min(1.0, (w - 2) * ss / max(1, tw))
        if sx >= 0.7 or size < 8: break
        size -= 1
    a = np.zeros((h, w))
    for i, t in enumerate(lines):
        b = f.getbbox(t)
        im = Image.new('L', (b[2] - b[0] + 2 * ss, desc - asc + 2 * ss), 0)
        ImageDraw.Draw(im).text((ss - b[0], ss - asc), t, font=f, fill=255)
        nw = max(1, int(im.width * sx / ss)); nh = max(1, int(im.height / ss))
        sm = np.asarray(im.resize((nw, nh), Image.LANCZOS)) / 255.0
        y0 = i * pitch + max(0, (pitch - nh) // 2)
        x0 = 0 if align == 'left' else max(0, (w - 1 - nw) // 2)
        hh, ww = min(nh, h - y0), min(nw, w - x0)
        if hh > 0 and ww > 0:
            a[y0:y0 + hh, x0:x0 + ww] = np.maximum(a[y0:y0 + hh, x0:x0 + ww], sm[:hh, :ww])
    return a

def colourise(a, fills, edge, dark, outline=False, bands=None):
    """fills: indices top->bottom for solid text; edge: antialias index;
    dark: shadow (or outline) index. bands: (y0, y1) row ranges, one per
    text line; the fill gradient then spans each line's glyph rows."""
    h, w = a.shape
    sh = np.zeros_like(a)
    if outline:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                s = np.roll(np.roll(a, dy, 0), dx, 1); sh = np.maximum(sh, s)
    else:
        sh[1:, 1:] = a[:-1, :-1]
    out = np.zeros((h, w), np.uint8)
    out[sh > 0.35] = dark
    out[a > 0.3] = edge
    solid = a > 0.55
    rows = np.array([fills[min(len(fills) - 1, y * len(fills) // h)] for y in range(h)], np.uint8)
    # gradient from the top of each line's glyphs, over the shortest line's
    # glyph height (descenders get the last colour), so every line gets the
    # same ramp whatever its position in the rectangle
    ext = [np.nonzero(solid[y0:y1].any(1))[0] for y0, y1 in bands or ()]
    n = min((len(e) and e[-1] - e[0] + 1 for e in ext if len(e)), default=0)
    for (y0, y1), ys in zip(bands or (), ext):
        if not len(ys): continue
        t = ys[0]
        for y in range(y0, y1):
            rows[y] = fills[min(len(fills) - 1, max(0, y - y0 - t) * len(fills) // n)]
    out[solid] = np.broadcast_to(rows[:, None], (h, w))[solid]
    return out

LABEL_FILLS = (0, 1, 2)                  # ranks in the yellow ramp, top->bottom

def label_img(lines, w, h, yellow, fills=LABEL_FILLS):
    """Mode label: yellow gradient per line, antialias edge, dark shadow."""
    pitch = h // len(lines)
    a = mask(lines, w, h, {1: 15, 2: 17}.get(len(lines), 14), pitch=pitch)
    bands = [(i * pitch, (i + 1) * pitch if i < len(lines) - 1 else h)
             for i in range(len(lines))]
    return colourise(a, [yellow[i] for i in fills], yellow[7], 15, bands=bands)

# Extras > Diary viewer help panel. NIKKI.BIN draws it as one sprite (u120
# v160 128x72, tpage 960, CLUT (448,480)) at screen (368,392). It lives in
# the MOJI2 font TIM (4bpp, VRAM (768,0), 1024x256 px), which exists as
# DAT2/MOJI2.TIZ and as a copy at MENU.BIZ 0x2f800; both get the same paint.
MOJI2_SRC = 'private/unpacked/MOJI2.TIZ.lz'
MOJI2_SLOT = 40 * 2048                 # disc allocation of MOJI2.TIZ
MENU_FONT_TIM = 0x2f800
HELP_RECT = (888, 160, 128, 72)        # panel in TIM pixels
HELP_FONT, HELP_SIZE = '/usr/share/fonts/TTF/RobotoCondensed-Bold.ttf', 15
# (tl id, x0, x1, baseline, cleared rows, last cleared col) in panel
# pixels. The circle/cross icons start at x 96 and the START mark at x 92;
# the JP text reaches x 89.
HELP_LINES = [('help_next', 6, 93, 17, (4, 25), 95), ('help_prev', 6, 93, 38, (25, 45), 95),
              ('help_back', 6, 88, 61, (45, 66), 90)]

def help_line(text, x0, x1, base, shape):
    """Alpha (shape) of one line, left at x0, squeezed to end by x1."""
    ss = 4
    f = ImageFont.truetype(HELP_FONT, HELP_SIZE * ss)
    b = f.getbbox(text); asc, desc = f.getmetrics()
    im = Image.new('L', (b[2] - b[0] + 4 * ss, asc + desc + 2 * ss), 0)
    ImageDraw.Draw(im).text((2 * ss - b[0], ss), text, font=f, fill=255)
    sx = min(1.0, (x1 - x0) * ss / (b[2] - b[0]))
    sm = np.asarray(im.resize((max(1, round(im.width * sx / ss)), round(im.height / ss)),
                              Image.LANCZOS)) / 255.0
    a = np.zeros(shape)
    oy, ox = round(base - 1 - asc / ss), round(x0 - 2 * sx)
    h, w = min(sm.shape[0], shape[0] - oy), min(sm.shape[1], shape[1] - ox)
    a[oy:oy + h, ox:ox + w] = sm[:h, :w]
    return a, sx

def help_panel(d, tim, tl):
    """Redraw the panel text in the font TIM at file offset `tim` (in place).
    White fill, 1 px dark outline, quantised to the panel's grey ramp
    (CLUT 12, indices 1-8); cleared areas get the 9/0 checker back."""
    pal = struct.unpack_from('<16H', d, tim + 20 + 12 * 32)
    greys = sorted(range(1, 9), key=lambda i: lum(pal[i]))
    lv = np.array([lum(pal[i]) * 255 / 186 for i in greys])
    x, y, w, h = HELP_RECT
    px = unpack(d, tim + 0x1c0, 1024, 256)
    pan = px[y:y + h, x:x + w]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = np.where((yy + xx) % 2 == 1, 9, 0).astype(np.uint8)
    squeeze = []
    for rid, x0, x1, base, (r0, r1), xc in HELP_LINES:
        pan[r0:r1, 4:xc + 1] = checker[r0:r1, 4:xc + 1]
        a, sx = help_line(tl[rid], x0, x1, base, (h, w))
        squeeze.append(sx)
        o = a.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                o = np.maximum(o, np.roll(np.roll(a, dy, 0), dx, 1))
        v = 16 * o * (1 - a) + 248 * a
        q = np.array(greys, np.uint8)[np.abs(v[..., None] - lv).argmin(-1)]
        m = o > 0.3
        m[:, xc + 1:] = False
        m[:r0] = m[r1:] = False
        pan[m] = q[m]
    pack(d, tim + 0x1c0, px)
    return squeeze

def wrap(text, width, size):
    f = ImageFont.truetype(FONT, size)
    lines = []
    for para in text.split('|'):
        cur = ''
        for word in para.split():
            t = (cur + ' ' + word).strip()
            if f.getlength(t) <= width - 3 or not cur: cur = t
            else: lines.append(cur); cur = word
        lines.append(cur)
    return lines

def build(png=False):
    tl = load_tl(); rc = rects()
    d = bytearray(open(SRC, 'rb').read())
    exe = bytearray(open(EXE, 'rb').read())
    t1 = unpack(d, T1, T1W, 256)
    t4 = unpack(d, T4, T4W, 2132)
    problems = 0
    prev = {}
    if png: os.makedirs('private/re/boot/en', exist_ok=True)

    # header: one string over hdr1 (256x40) and hdr2 (112x40)
    pal = palette(d, 464, 481)
    x1, y1, w1, h1 = rc['hdr1']; x2, y2, w2, h2 = rc['hdr2']
    used = sorted({int(i) for i in np.unique(t1[y1:y1+h1, x1:x1+w1]) if i}, key=lambda i: lum(pal[i]))
    a = mask([tl['hdr']], w1 + w2, h1, 30, font=FONTH)
    img = colourise(a, used[-1:-6:-1][::-1][::-1], used[len(used) // 2], used[0], outline=True)
    t1[y1:y1+h1, x1:x1+w1] = img[:, :w1]; t1[y2:y2+h2, x2:x2+w2] = img[:, w1:]
    prev['hdr'] = (img, pal)

    # labels: yellow gradient (5-14) with a dark drop shadow (15)
    pal = palette(d, 448, 481)
    yellow = sorted(range(5, 15), key=lambda i: -lum(pal[i]))
    for k in range(13):
        rid = f'label{k}'
        x, y, w, h = rc[rid]
        lines = tl[rid].split('|')
        img = label_img(lines, w, h, yellow)
        t1[y:y+h, x:x+w] = img
        prev[rid] = (img, pal)

    # descriptions
    pal = palette(d, 16, 482)
    fills = sorted(range(1, 9), key=lambda i: -lum(pal[i]))
    darks = sorted(range(9, 16), key=lambda i: lum(pal[i]))
    for m in range(13):
        p = F(struct.unpack_from('<I', exe, F(LISTS) + 4 * m)[0])
        n = struct.unpack_from('<h', exe, p)[0]
        spr = [list(struct.unpack_from('<11h', exe, p + 2 + 22 * k)) for k in range(n)]
        text = [s for s in spr if s[5] == 0 and s[6] == 0][0]
        tail = [s for s in spr if s[5] == 0 and s[6] > 0]
        width = 255 + (96 if tail else 0)
        lines = wrap(tl[f'desc{m}'], width, 18)
        if len(lines) > 3: print(f'desc{m}: {len(lines)} lines'); problems += 1; lines = lines[:3]
        a = mask(lines, width, 80, 18, pitch=24)
        img = colourise(a, fills[:3], fills[5], darks[0])
        blk = t4[m * 164:(m + 1) * 164]
        blk[0:80, 0:255] = img[:, :255]
        blk[0:80, 255] = 0
        text[7], text[8] = 255, 80
        if tail:
            blk[80:160, 0:96] = img[:, 255:351]
            t = tail[0]; t[6], t[7], t[8] = 80, 96, 80
        for k, s in enumerate(spr):
            struct.pack_into('<11h', exe, p + 2 + 22 * k, *s)
        prev[f'desc{m}'] = (img, pal)

    pack(d, T1, t1); pack(d, T4, t4)

    # Extras help panel, in both copies of the MOJI2 font TIM
    m2 = bytearray(open(MOJI2_SRC, 'rb').read())
    assert d[MENU_FONT_TIM:MENU_FONT_TIM + 0x1c0] == m2[:0x1c0], 'MENU.BIZ font TIM moved'
    for buf, off in ((d, MENU_FONT_TIM), (m2, 0)):
        sq = help_panel(buf, off, tl)
    for (rid, *_), sx in zip(HELP_LINES, sq):
        if sx < 0.85: print(f'{rid}: squeezed to {sx:.2f}'); problems += 1
    c2 = lz.compress(bytes(m2))
    if len(c2) > MOJI2_SLOT: print(f'MOJI2.TIZ: {len(c2)} bytes > slot {MOJI2_SLOT}'); problems += 1
    open('private/build/MOJI2.TIZ.lz', 'wb').write(m2)
    open('private/build/MOJI2.TIZ', 'wb').write(c2)
    if png:
        x, y, w, h = HELP_RECT
        pal = struct.unpack_from('<16H', m2, 20 + 12 * 32)
        prev['help'] = (unpack(m2, 0x1c0, 1024, 256)[y:y + h, x:x + w], pal)
    if png:
        for rid, (img, pal) in prev.items():
            rgb = np.array([[(c & 31) * 8, (c >> 5 & 31) * 8, (c >> 10 & 31) * 8] for c in pal], np.uint8)
            Image.fromarray(rgb[img]).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(
                f'private/re/boot/en/{rid}.png')
    c = lz.compress(bytes(d))
    if len(c) > SLOT: print(f'MENU.BIZ: {len(c)} bytes > slot {SLOT}'); problems += 1
    open('private/build/MENU.BIZ.lz', 'wb').write(d)
    open('private/build/MENU.BIZ', 'wb').write(c)
    open(EXE, 'wb').write(exe)
    print(f'MENU: compressed {len(c)} bytes (slot {SLOT}), MOJI2 {len(c2)} (slot {MOJI2_SLOT}), '
          f'help squeeze {" ".join(f"{s:.2f}" for s in sq)}, problems {problems}')

if __name__ == '__main__':
    build('--png' in sys.argv)
