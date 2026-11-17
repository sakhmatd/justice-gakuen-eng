# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Redraw the character-creation graphics (DAT2/MA_ENT.BIZ) in English:
keyboard tabs and side buttons, the 入学願書 header and the vertical school
names. Reads tl/ENTRY.txt (ID: EN). Writes private/build/MA_ENT.BIZ.
--png writes previews to private/re/entry/en/.

MA_ENT.BIZ is one LZ stream: a 4bpp 1024x256 TIM at VRAM (512,256), 20
CLUTs at VRAM (0,483) (file 0x14), pixels at 0x2a0 (512 bytes per row).
Rectangles and palettes come from private/re/entry/labels.tsv (see the RE
notes there). The builder keeps each label's own background colour,
paints the old text out with it and draws the English with the label's
main text colour, a mid tone for edges and its darkest colour as outline.
"""
import sys, os, struct, collections
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import lz

SRC = 'private/unpacked/MA_ENT.BIZ.lz'
PIX, STRIDE = 0x2a0, 512
CLUT = 0x14
FONT = '/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf'
FONT_HDR = '/usr/share/fonts/TTF/DejaVuSerif.ttf'
FONT_SCH = '/usr/share/fonts/TTF/ariblk.ttf'
# id: (texel x, y, w, h, palette, kind)
RECTS = {
    'tab_hira':  (416, 160, 48, 24, 17, 'label'),
    'tab_kata':  (464, 160, 48, 24, 17, 'label'),
    'tab_abc':   (416, 184, 48, 24, 17, 'label'),
    'tab_other': (464, 184, 48, 24, 17, 'label'),
    'btn_del':   (416, 208, 48, 24, 18, 'label'),
    'btn_next':  (464, 208, 48, 24, 18, 'label'),
    'btn_back':  (416, 232, 48, 24, 18, 'label'),
    'btn_ok':    (464, 232, 48, 24, 16, 'label'),
    'header':    (568, 160, 136, 24, 10, 'header'),
    'sch_taiyo':   (928, 0, 32, 112, 9, 'school'),
    'sch_gorin':   (896, 0, 32, 112, 8, 'school'),
    'sch_pacific': (864, 0, 32, 152, 7, 'school'),
    'sch_gedo':    (832, 0, 32, 112, 6, 'school'),
}

def load_tl(path='tl/ENTRY.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

def lum(v): return (v & 31) * 2 + (v >> 5 & 31) * 3 + (v >> 10 & 31)

def text_alpha(text, w, h, font, size, ss=4, rotate=False):
    if rotate: w, h = h, w
    while True:
        f = ImageFont.truetype(font, size * ss)
        b = f.getbbox(text)
        if b[2] - b[0] <= (w - 3) * ss or size <= 7: break
        size -= 1
    im = Image.new('L', (w * ss, h * ss), 0)
    tw, th = b[2] - b[0], b[3] - b[1]
    ImageDraw.Draw(im).text(((w * ss - tw) // 2 - b[0], (h * ss - th) // 2 - b[1]), text,
                            font=f, fill=255)
    if rotate: im = im.rotate(-90, expand=True)
    im = im.resize((im.width // ss, im.height // ss), Image.LANCZOS)
    return np.asarray(im, float) / 255

def build(png=False):
    tl = load_tl()
    d = bytearray(open(SRC, 'rb').read())
    clut = struct.unpack_from('<320H', d, CLUT)
    raw = np.frombuffer(bytes(d[PIX:PIX + STRIDE * 256]), np.uint8).reshape(256, STRIDE)
    px = np.empty((256, 1024), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    prev = {}
    for rid, (x, y, w, h, p, kind) in RECTS.items():
        if rid not in tl: continue
        pc = clut[p * 16:p * 16 + 16]
        sub = px[y:y + h, x:x + w]
        cnt = collections.Counter(sub.ravel().tolist())
        border = np.concatenate([sub[0], sub[-1], sub[:, 0], sub[:, -1]])
        bg = collections.Counter(border.tolist()).most_common(1)[0][0]
        others = [i for i, n in cnt.most_common() if i != bg and n > 3]
        main = others[0]
        dark = min(others, key=lambda i: lum(pc[i]))
        light = max(others, key=lambda i: lum(pc[i]))
        mid = sorted(others, key=lambda i: lum(pc[i]))[len(others) // 2]
        out = sub.copy()
        if kind == 'header':
            # keep the underline: rows where most texels are not background
            line_rows = [r for r in range(h) if (sub[r] != bg).mean() > 0.6]
            top = min(line_rows) if line_rows else h
            out[:top][out[:top] != bg] = bg
            a = text_alpha(tl[rid], w, top, FONT_HDR, 15)
            region = out[:top]
            # strokes use indices 8-15, their light antialias 1-4
            stroke = max((i for i in others if i >= 8), key=lambda i: cnt[i], default=main)
            soft = max((i for i in others if 1 <= i <= 4), key=lambda i: cnt[i], default=mid)
            region[(a > 0.25) & (a <= 0.6)] = soft; region[a > 0.6] = stroke
        else:
            # clear old text, keep the label background
            out[out != bg] = bg
            if kind == 'school':
                a = text_alpha(tl[rid], w, h, FONT_SCH, 22, rotate=True)
                ol = 1
                fill = sorted(others, key=lambda i: -cnt[i])[0]
            else:
                a = text_alpha(tl[rid], w, h, FONT, 14)
                ol = 0
                fill = light
            o = a.copy()
            st = ndimage.generate_binary_structure(2, 2 if ol else 1)
            if ol: o = ndimage.grey_dilation(o, footprint=st)
            sh = np.zeros_like(a); sh[1:, 1:] = o[:-1, :-1]
            out[np.maximum(sh, o if ol else 0) > 0.4] = dark
            out[a > 0.3] = mid
            out[a > 0.55] = fill
        px[y:y + h, x:x + w] = out
        prev[rid] = (out, pc)
    d[PIX:PIX + STRIDE * 256] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()
    c = lz.compress(bytes(d))
    slot = os.path.getsize('private/disc/DAT2/MA_ENT.BIZ'); slot += -slot % 2048
    if len(c) > slot: print(f'MA_ENT: {len(c)} > slot {slot}'); return 1
    open('private/build/MA_ENT.BIZ.lz', 'wb').write(d)
    open('private/build/MA_ENT.BIZ', 'wb').write(c)
    if png:
        os.makedirs('private/re/entry/en', exist_ok=True)
        for rid, (img, pc) in prev.items():
            rgb = np.array([[(v & 31) * 8, (v >> 5 & 31) * 8, (v >> 10 & 31) * 8] for v in pc], np.uint8)
            Image.fromarray(rgb[img]).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST).save(
                f'private/re/entry/en/{rid}.png')
    print(f'MA_ENT: {len(prev)} labels, compressed {len(c)} bytes (slot {slot})')
    return 0

if __name__ == '__main__':
    build('--png' in sys.argv)
