# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Add the translation credit to the title screen (DAT2/TITLE.BIZ).

TITLE.BIZ (decompressed) holds one 8bpp TIM at 0x6000: CLUT (512,480)
256x1, pixels 512x480 at VRAM (512,0), shown 1:1 on the 512x480 title
screen. The "(C)CAPCOM ..." line is part of this picture (ink rows
432-445, x 26-480). The credit is drawn into the picture below it, in the
picture's own CLUT (nearest colour), and the file is recompressed from the
original disc copy, so reruns give the same output.

Writes private/build/TITLE.BIZ (+ .lz). No exe change; independent of the
other build scripts.
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import lz, burst

VERSION = '1.1'                         # release version shown in the credit
YEAR = '2026'
CREDIT = f'English translation by sakhmatd - v{VERSION} - {YEAR}'

SRC = 'private/unpacked/TITLE.BIZ.lz'
SLOT = 88 * 2048                        # disc allocation of TITLE.BIZ
TIM = 0x6000
CLUT = TIM + 20                         # 256 entries
PIX = 0x6220                            # 512 bytes per row, 480 rows
W, H = 512, 480
ARIAL = '/usr/share/fonts/TTF/arialbd.ttf'

# Variants (the preview script in private/re/titlescreen/ draws them all).
# x: 'c' centres on the copyright line (ink x 26-480), 'r' ends at x 480.
# top: first row of cap height (copyright caps start at row 433).
STYLES = {
    'v1': dict(text=CREDIT, size=11, x='c', top=453, fg=168, shadow=0.8),
    'v2': dict(text=f'TL by sakhmatd v{VERSION} ({YEAR})', size=11, x='r',
               top=453, fg=168, shadow=0.8),
    'v3': dict(text=f'v{VERSION} EN by sakhmatd', size=11, x='r', top=453,
               fg=168, shadow=0.8),
    'v4': dict(text=CREDIT, size=11, x='c', top=453, fg=150, shadow=0.0),
}
DEFAULT = 'v3'
CR_X0, CR_X1 = 26, 480                  # copyright ink extent


def load():
    d = bytearray(open(SRC, 'rb').read())
    assert struct.unpack_from('<II', d, TIM) == (0x10, 9)
    assert struct.unpack_from('<IHHHH', d, PIX - 12) == (12 + W * H, 512, 0, 256, 480)
    clut = np.frombuffer(bytes(d[CLUT:CLUT + 512]), '<u2')
    pal = np.stack([(clut & 31) << 3, (clut >> 5 & 31) << 3, (clut >> 10 & 31) << 3], -1)
    px = np.frombuffer(bytes(d[PIX:PIX + W * H]), np.uint8).reshape(H, W).copy()
    return d, pal.astype(float), px


def text_mask(text, size, font=ARIAL):
    """Anti-aliased coverage (0..1) cropped to the ink, and the cap top row
    offset inside the crop."""
    f = ImageFont.truetype(font, size)
    m = Image.new('L', (W, 3 * size)); dr = ImageDraw.Draw(m)
    dr.text((4, size), text, font=f, fill=255)
    a = np.array(m, float) / 255
    ys, xs = np.where(a > 0)
    cap = Image.new('L', (W, 3 * size)); ImageDraw.Draw(cap).text((4, size), 'H', font=f, fill=255)
    cy = np.where(np.array(cap) > 127)[0].min()
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1], int(cy - ys.min())


def paint(pal, px, text, size, x, top, fg, shadow, font=ARIAL):
    """Draw text into the 8bpp picture px (in place) with nearest CLUT colours."""
    a, capoff = text_mask(text, size, font)
    h, w = a.shape
    x0 = (CR_X0 + CR_X1 + 1 - w) // 2 if x == 'c' else CR_X1 + 1 - w if x == 'r' else x
    y0 = top - capoff
    # 1 px drop shadow down-right, then the grey text over it.
    hh, ww = h + 1, w + 1
    A = np.zeros((hh, ww)); A[:h, :w] = a
    Sh = np.zeros((hh, ww)); Sh[1:, 1:] = a
    reg = px[y0:y0 + hh, x0:x0 + ww]
    col = pal[reg] * (1 - shadow * Sh[..., None])
    col = col * (1 - A[..., None]) + fg * A[..., None]
    touched = (A > 0) | (Sh * shadow > 0)
    d2 = ((col[..., None, :] - pal[None, None]) ** 2).sum(-1)
    new = d2.argmin(-1).astype(np.uint8)
    reg[touched] = new[touched]
    return (x0, y0, ww, hh)


def build(style=DEFAULT):
    d, pal, px = load()
    box = paint(pal, px, **STYLES[style])
    d[PIX:PIX + W * H] = px.tobytes()
    c = lz.compress(bytes(d), chain=4096, burst=0x1600)
    out, _ = lz.decompress(c)
    assert out == bytes(d)
    peak = max(burst.bursts(c).values())
    problems = 0
    if len(c) > SLOT: print(f'TITLE.BIZ: {len(c)} bytes > slot {SLOT}'); problems += 1
    if peak > 0x1600: print(f'TITLE.BIZ: burst {peak:#x} > 0x1600'); problems += 1
    open('private/build/TITLE.BIZ.lz', 'wb').write(d)
    open('private/build/TITLE.BIZ', 'wb').write(c)
    print(f'TITLE: "{STYLES[style]["text"]}" at {box}, compressed {len(c)} bytes '
          f'(slot {SLOT}), burst {peak:#x}, {problems} problems')
    return problems


if __name__ == '__main__':
    sys.exit(1 if build(sys.argv[1] if len(sys.argv) > 1 else DEFAULT) else 0)
