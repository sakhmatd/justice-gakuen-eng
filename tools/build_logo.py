# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""English diary-mode logo, replacing the 熱血青春日記 bitmap.

Design: "HOT-BLOODED" in heavy italic (Impact, sheared) with a red ->
yellow fire gradient, a 1 px top highlight, a dark outline and a 3 px drop
shadow. "YOUTH DIARY" is a tracked tag in yellow on a slanted dark-red bar
under the right end of the main word.

Two copies are written (private/re/logo/ has the analysis):
- Enroll/Continue screen: DAT2/MA_TIT.BIZ, 4bpp TIM at 0x14800, 256x80
  (pixels at 0x14a20, 128 bytes per row), drawn 1:1 at (32,32).
- Hub top-right card: DAT2/MA_COCK.BIZ TIM1, tpage 704 u0-255 v176-255
  (TIM x 768-1023). The original drew this texture shrunk to 128x40 at
  (376,72), over the card's bottom border. The draw code in MGAME and MGFIN
  is patched to draw it 1:1 as 256x80 at (248,32), and the design is
  centred in the card's inner box (screen x 259-493, y 35-109). Run after
  build_menus.py (it rewrites private/build/MA_COCK.BIZ from that tool's
  output) and after hook.py (MGAME.BIN, MGFIN.BIN).

Palette (both copies, not changed): 1 near black, 2-5 dark reds, 7-15 the
fill ramp red -> yellow.
--png writes previews to private/re/logo/en/.
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import lz

IMPACT = '/usr/share/fonts/TTF/impact.ttf'
MAIN, TAG = 'HOT-BLOODED', 'YOUTH DIARY'
SS = 8                                  # supersampling for text masks
SLANT = 0.25
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 - 0.5
DITH = 0.5

# Hub draw code: (file offset, original word, new word). Quad size is a
# double whose high word is loaded with lui; position is set at init.
HUB_X, HUB_Y = 248, 32
CODE = {
    'MGAME.BIN': [(0x17ff8, 0x3c154060, 0x3c154070),      # w 128.0 -> 256.0
                  (0x18068, 0x3c134044, 0x3c134054),      # h 40.0 -> 80.0
                  (0x17950, 0x24020178, 0x24020000 | HUB_X),
                  (0x17958, 0x24020048, 0x24020000 | HUB_Y)],
    'MGFIN.BIN': [(0xb284, 0x3c154060, 0x3c154070),
                  (0xb2f4, 0x3c134044, 0x3c134054),
                  (0xabdc, 0x24020178, 0x24020000 | HUB_X),
                  (0xabe4, 0x24020048, 0x24020000 | HUB_Y)],
}
# Ink boxes (x0, y0, x1, y1) inside the 256x80 texture.
HUB_BOX = (14, 9, 242, 76)              # card inner box minus a margin
ENROLL_BOX = (0, 0, 252, 78)

def text_hi(t, cap, track=0.0):
    """Supersampled mask of t, cap height cap px, tracking in cap units."""
    size = 10
    while True:
        f = ImageFont.truetype(IMPACT, size)
        b = f.getbbox('H')
        if b[3] - b[1] >= cap * SS: break
        size += 2
    hb = f.getbbox('H')
    adv = [f.getlength(c) for c in t]
    tw = int(sum(adv) + track * cap * SS * (len(t) - 1)) + 4 * SS
    im = Image.new('L', (tw, cap * SS + 4 * SS), 0)
    dr = ImageDraw.Draw(im)
    x = 2 * SS
    for c, a in zip(t, adv):
        dr.text((x, 2 * SS - hb[1]), c, font=f, fill=255)
        x += a + track * cap * SS
    return im.crop(im.getbbox())

def shear(im, k):
    W, H = im.size
    out = Image.new('L', (W + int(k * H) + 2, H), 0)
    out.paste(im, (int(k * H), 0))
    return out.transform(out.size, Image.AFFINE, (1, k, 0, 0, 1, 0), Image.BICUBIC)

def down(im, W, H):
    return np.asarray(im.resize((W, H), Image.LANCZOS), float) / 255

def ramp(shape, r0, r1, lo, hi, flip=False):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    t = np.clip((yy - r0) / max(1, r1 - r0), 0, 1)
    if flip: t = 1 - t
    k = lo + t * (hi - lo) + BAYER[yy % 4, xx % 4] * DITH
    return np.clip(np.rint(k), lo, hi).astype(np.uint8)

def design(box, cap1, cap2, W=256, H=80):
    x0, y0, x1, y1 = box
    # main word, stretched to the box width
    t1 = shear(text_hi(MAIN, cap1), SLANT)
    w1 = x1 - x0 - 6
    h1 = int(t1.size[1] * w1 / t1.size[0])
    barh = cap2 + 6
    ty = y0 + (y1 - y0 - (h1 + 2 + barh + 4)) // 2 + 1
    a1 = np.zeros((H, W))
    a1[ty:ty + h1, x0 + 2:x0 + 2 + w1] = down(t1, w1, h1)
    fill1 = a1 >= 0.5
    edge1 = (a1 >= 0.15) & ~fill1
    # tag on a slanted bar, right-aligned under the main word
    t2 = shear(text_hi(TAG, cap2, track=0.28), SLANT)
    w2, h2 = t2.size[0] // SS, t2.size[1] // SS
    by = ty + h1 + 2
    bx1 = x1 - 4
    bx0 = bx1 - w2 - 14
    bar = np.zeros((H, W), bool)
    for y in range(by, by + barh):
        off = int((by + barh - y) * SLANT)
        bar[y, bx0 + off:bx1 + off - int(barh * SLANT)] = True
    a2 = np.zeros((H, W))
    tx, tyy = bx0 + 7, by + (barh - h2) // 2
    a2[tyy:tyy + h2, tx:tx + w2] = down(t2, w2, h2)
    fill2 = (a2 >= 0.5) & bar
    # layers: shadow, bar, outline, main fill, tag fill
    out = np.zeros((H, W), np.uint8)
    body = fill1 | edge1
    ol = ndimage.binary_dilation(body, np.ones((3, 3))) & ~body
    shape = body | ol | bar
    sh = np.zeros_like(shape); sh[3:, 3:] = shape[:-3, :-3]
    out[sh] = 1
    out[bar] = 2
    out[bar & ~np.roll(bar, 1, 0)] = 4                  # bar top edge
    out[(a2 >= 0.2) & bar & ~fill2] = 3
    out[ol] = 1
    out[edge1] = 2
    rows = np.where(fill1.any(1))[0]
    g = ramp(fill1.shape, rows[0], rows[-1], 7, 15)
    out[fill1] = g[fill1]
    hl = fill1 & ~np.roll(fill1, 1, 0)                  # top highlight
    out[hl] = np.minimum(15, g[hl] + 4)
    r = np.where(fill2.any(1))[0]
    g2 = ramp(fill2.shape, r[0], r[-1], 13, 15, flip=True)
    out[fill2] = g2[fill2]
    return out

def put4(d, off, stride, x, y, img):
    """Write a 4bpp index image into TIM pixel data."""
    h, w = img.shape
    for r in range(h):
        o = off + (y + r) * stride
        row = bytearray(d[o:o + stride])
        for c in range(w):
            X = x + c; b = row[X // 2]
            row[X // 2] = (b & 0xf0) | img[r, c] if X % 2 == 0 else (b & 0x0f) | (img[r, c] << 4)
        d[o:o + stride] = row

def pal(d, off):
    c = struct.unpack_from('<16H', d, off)
    return np.array([[(v & 31) * 8, (v >> 5 & 31) * 8, (v >> 10 & 31) * 8] for v in c], np.uint8)

def patch_code():
    for name, edits in CODE.items():
        p = 'private/build/' + name
        mg = bytearray(open(p, 'rb').read())
        for off, old, new in edits:
            w = struct.unpack_from('<I', mg, off)[0]
            if w not in (old, new): sys.exit(f'{name} {off:#x}: {w:#010x} is not the logo draw code')
            struct.pack_into('<I', mg, off, new)
        open(p, 'wb').write(mg)

def build(png=False):
    big = design(ENROLL_BOX, 36, 13)
    hub = design(HUB_BOX, 30, 11)
    d = bytearray(open('private/unpacked/MA_TIT.BIZ.lz', 'rb').read())
    put4(d, 0x14a20, 128, 0, 0, big)
    c = lz.compress(bytes(d))
    slot = os.path.getsize('private/disc/DAT2/MA_TIT.BIZ')
    slot += -slot % 2048
    if len(c) > slot: print(f'MA_TIT: {len(c)} > {slot}'); return 1
    m = bytearray(open('private/build/MA_COCK.BIZ.lz', 'rb').read())
    put4(m, 0x2020, 512, 768, 176, hub)
    c2 = lz.compress(bytes(m))
    slot2 = os.path.getsize('private/disc/DAT2/MA_COCK.BIZ')
    slot2 += -slot2 % 2048
    if len(c2) > slot2: print(f'MA_COCK: {len(c2)} > {slot2}'); return 1
    open('private/build/MA_TIT.BIZ.lz', 'wb').write(d)
    open('private/build/MA_TIT.BIZ', 'wb').write(c)
    open('private/build/MA_COCK.BIZ.lz', 'wb').write(m)
    open('private/build/MA_COCK.BIZ', 'wb').write(c2)
    patch_code()
    if png:
        os.makedirs('private/re/logo/en', exist_ok=True)
        p = pal(d, 0x14814)
        for name, img, bg in (('enroll', big, (40, 40, 80)), ('hub', hub, (16, 16, 120))):
            rgb = np.zeros(img.shape + (3,), np.uint8); rgb[:] = bg
            rgb[img > 0] = p[img][img > 0]
            Image.fromarray(rgb).resize((768, 240), Image.NEAREST).save(f'private/re/logo/en/{name}.png')
    print(f'logo: MA_TIT {len(c)} bytes (slot {slot}), MA_COCK {len(c2)} bytes (slot {slot2})')
    return 0

if __name__ == '__main__':
    sys.exit(build('--png' in sys.argv))
