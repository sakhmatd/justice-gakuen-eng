# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Festival (五輪祭) mode menus in English.

- Label atlas: TIM0 at 0 of every DAT2/MINITTL0-8.BIZ (identical copies),
  4bpp 768x256 at VRAM (768,0), 9 CLUTs at (0,481) (file 0x14), pixels at
  0x140 (384 bytes per row). Holds the four game names, "back to mode
  menu", the minigame title-menu items and the record labels.
- Header 五輪祭: painted into the 8bpp 512x480 background of MINITTL0 (TIM at
  0x18800, CLUT +0x14, pixels +0x220). Repainted like the event banners,
  with the original's left-to-right colour gradient.
Reads tl/FEST.txt. Writes private/build/MINITTL0-8.BIZ. --png previews.
Rectangles: private/re/fest/fest.tsv.
"""
import sys, os, struct, collections
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from scipy import ndimage
import lz
from build_title import mask as text_mask_lines
import build_banners as BB

PIX, STRIDE, W, H = 0x140, 384, 768, 256
# id: (x, y, w, h, clut)
ATLAS = {
    'item_home_run': (0, 0, 200, 24, 0), 'item_pk': (0, 24, 152, 24, 0),
    'item_shoot': (0, 48, 216, 24, 0), 'item_ace': (0, 72, 184, 24, 0),
    'item_back': (512, 0, 208, 24, 0),
    'fest_record': (0, 192, 120, 24, 2), 'unit_pt': (160, 216, 16, 24, 2),
    'title_start': (0, 96, 144, 24, 1), 'title_back': (144, 96, 40, 24, 1),
    'title_1p': (0, 120, 160, 24, 1), 'title_vs': (0, 144, 160, 24, 1),
    'title_rules': (0, 168, 160, 24, 1), 'title_record': (0, 216, 160, 24, 2),
}
HEADER = (150, 14, 364, 108)            # x0, y0, x1, y1 in the MINITTL0 bg

def load_tl(path='tl/FEST.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

def lum(v): return (v & 31) * 2 + (v >> 5 & 31) * 3 + (v >> 10 & 31)

def atlas(d, tl, prev):
    clut = struct.unpack_from('<144H', d, 0x14)
    raw = np.frombuffer(bytes(d[PIX:PIX + STRIDE * H]), np.uint8).reshape(H, STRIDE)
    px = np.empty((H, W), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    for rid, (x, y, w, h, c) in ATLAS.items():
        if rid not in tl: continue
        pc = clut[c * 16:c * 16 + 16]
        sub = px[y:y + h, x:x + w]
        cnt = collections.Counter(sub.ravel().tolist())
        used = [i for i, n in cnt.items() if i and n > 2]
        ramp = sorted(used, key=lambda i: lum(pc[i]))
        main = max((i for i in used if i != ramp[0]), key=lambda i: cnt[i])
        a = text_mask_lines([tl[rid]], w, h, 16)
        o = ndimage.grey_dilation(a, footprint=np.ones((3, 3)))
        sh = np.zeros_like(a); sh[1:, 1:] = o[:-1, :-1]
        out = np.zeros((h, w), np.uint8)
        out[np.maximum(o, sh) > 0.4] = ramp[0]
        edge = sorted(used, key=lambda i: abs(lum(pc[i]) - (lum(pc[main]) + lum(pc[ramp[0]])) / 2))[0]
        out[a > 0.3] = edge
        out[a > 0.55] = main
        px[y:y + h, x:x + w] = out
        prev[rid] = (out, pc)
    d[PIX:PIX + STRIDE * H] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()

def header(d, text):
    t = 0x18800
    clut = struct.unpack_from('<256H', d, t + 0x14)
    pal = np.array([BB.rgb15(v) for v in clut])
    idx = np.frombuffer(bytes(d[t + 0x220:t + 0x220 + 512 * 480]), np.uint8).reshape(480, 512).copy()
    img = pal[idx]
    x0, y0, x1, y1 = HEADER
    mk, fill, line, raw = BB.text_mask(img, x0, y0, x1, y1)
    # per-column fill colour of the original (left -> right gradient)
    core = ndimage.distance_transform_edt(raw) >= 2
    sat = img.max(-1) - img.min(-1)
    def colcol(x):
        sel = core[:, x]
        if not sel.any(): return None
        c = img[sel, x]; s_ = sat[sel, x]
        return c[s_ >= np.percentile(s_, 60)].mean(0)
    cols = [colcol(x) for x in range(x0, x1)]
    # outline: the darkest quarter of the stroke-edge pixels
    edge = raw & ~ndimage.binary_erosion(raw)
    e = img[raw]; line = e[e.sum(-1) <= np.percentile(e.sum(-1), 5)].mean(0)
    cols = [c for c in cols if c is not None]
    img = BB.inpaint(img, mk)
    # draw: outline in the original outline colour, fill with the gradient
    lines = text.split('|')
    a = text_mask_lines(lines, x1 - x0, y1 - y0, 40, font=BB.FONT_T, align='centre',
                        pitch=(y1 - y0) // len(lines))
    o = a.copy()
    for _ in range(2): o = ndimage.grey_dilation(o, footprint=np.ones((3, 3)))
    sh = ndimage.shift(o, (4, 4), order=0) * 0.7
    reg = img[y0:y1, x0:x1]
    reg[:] = reg * (1 - sh[..., None]) + line * sh[..., None]
    reg[:] = reg * (1 - o[..., None]) + line * o[..., None]
    xs = np.where(a.max(0) > 0.3)[0]
    for x in range(a.shape[1]):
        if not len(xs): break
        t_ = (x - xs[0]) / max(1, xs[-1] - xs[0])
        c = cols[min(len(cols) - 1, max(0, int(t_ * (len(cols) - 1))))]
        reg[:, x] = reg[:, x] * (1 - a[:, x, None]) + c * a[:, x, None]
    new = BB.quantize(np.clip(img, 0, 255), pal, idx)
    d[t + 0x220:t + 0x220 + 512 * 480] = new.tobytes()
    return pal[new]

def build(png=False):
    tl = load_tl(); prev = {}; problems = 0
    for k in range(9):
        n = f'MINITTL{k}'
        d = bytearray(open(f'private/unpacked/{n}.BIZ.lz', 'rb').read())
        atlas(d, tl, prev)
        if k == 0 and 'header' in tl:
            hdr = header(d, tl['header'])
            if png: Image.fromarray(hdr.astype(np.uint8)).save('private/re/fest/en_header.png')
        c = lz.compress(bytes(d))
        slot = os.path.getsize(f'private/disc/DAT2/{n}.BIZ'); slot += -slot % 2048
        if len(c) > slot: print(f'{n}: {len(c)} > {slot}'); problems += 1; continue
        open(f'private/build/{n}.BIZ.lz', 'wb').write(d)
        open(f'private/build/{n}.BIZ', 'wb').write(c)
    if png:
        os.makedirs('private/re/fest/en', exist_ok=True)
        for rid, (img, pc) in prev.items():
            rgb = np.array([[(v & 31) * 8, (v >> 5 & 31) * 8, (v >> 10 & 31) * 8] for v in pc], np.uint8)
            Image.fromarray(rgb[img]).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(
                f'private/re/fest/en/{rid}.png')
    print(f'MINITTL: {len(prev)} labels, problems {problems}')

if __name__ == '__main__':
    build('--png' in sys.argv)
