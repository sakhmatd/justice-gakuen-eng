# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""English logo tile in every copy of the 144x96 私立ジャスティス学園 tile.

The tile (4bpp, 7 colour schemes) is byte-identical in several files. The
drawing is build_examgfx.logo_tile(), which already handles the SIKEN2
copy. This tool patches the other copies:
- MENU.BIZ 0x50000 (after build_title.py, reads private/build/MENU.BIZ.lz)
- MZBG03, MZBG07, MZBG12 archive entry 0 at 0x1b000
- MZBG13.BIZ 0x26800, SELP3.BIZ 0x6800
MZPANEL entries 0-2 (0x41000) are done by build_diarygfx.py.
Writes private/build/NAME (and NAME.lz for plain files).
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import lz
from build_examgfx import logo_tile

PLAIN = [('MENU.BIZ', 0x50000), ('MZBG13.BIZ', 0x26800), ('SELP3.BIZ', 0x6800)]
ARCS = [('MZBG03.BIZ', 0, 0x1b000), ('MZBG07.BIZ', 0, 0x1b000), ('MZBG12.BIZ', 0, 0x1b000)]

def tim_px(d, off):
    """(pixel offset, width in pixels, height) of a 4bpp TIM."""
    assert struct.unpack_from('<I', d, off)[0] == 0x10, hex(off)
    flags = struct.unpack_from('<I', d, off + 4)[0]
    assert flags & 7 == 0, 'not 4bpp'
    p = off + 8
    if flags & 8: p += struct.unpack_from('<I', d, p)[0]
    bl, x, y, w, h = struct.unpack_from('<IHHHH', d, p)
    return p + 12, w * 4, h

def patch(d, off):
    p, w, h = tim_px(d, off)
    assert (w, h) == (144, 96), (w, h)
    raw = np.frombuffer(bytes(d[p:p + w * h // 2]), np.uint8).reshape(h, w // 2)
    px = np.empty((h, w), np.uint8); px[:, 0::2] = raw & 15; px[:, 1::2] = raw >> 4
    logo_tile(px)
    d[p:p + w * h // 2] = (px[:, 0::2] | (px[:, 1::2] << 4)).astype(np.uint8).tobytes()

def build():
    problems = 0
    for name, off in PLAIN:
        src = f'private/build/{name}.lz'
        d = bytearray(open(src if os.path.exists(src) else f'private/unpacked/{name}.lz', 'rb').read())
        patch(d, off)
        c = lz.compress(bytes(d))
        slot = os.path.getsize(f'private/disc/DAT2/{name}'); slot += -slot % 2048
        if len(c) > slot: print(f'{name}: {len(c)} > {slot}'); problems += 1; continue
        open(src, 'wb').write(d)
        open(f'private/build/{name}', 'wb').write(c)
        print(f'{name}: {len(c)} bytes (slot {slot})')
    for name, e, off in ARCS:
        bp = f'private/build/{name}'
        arc = bytearray(open(bp if os.path.exists(bp) else f'private/disc/DAT2/{name}', 'rb').read())
        o, s = struct.unpack_from('<II', arc, 8 * e)
        o1 = struct.unpack_from('<I', arc, 8 * (e + 1))[0] or len(arc)
        d = bytearray(open(f'private/unpacked/{name}/{e:03d}.lz', 'rb').read())
        patch(d, off)
        c = lz.compress(bytes(d))
        if len(c) > o1 - o: print(f'{name} {e}: {len(c)} > {o1 - o}'); problems += 1; continue
        arc[o:o1] = c + bytes(o1 - o - len(c))
        struct.pack_into('<I', arc, 8 * e + 4, len(c))
        open(bp, 'wb').write(arc)
        print(f'{name} entry {e}: {len(c)} bytes (slot {o1 - o})')
    print(f'logotile: problems {problems}')
    return problems

if __name__ == '__main__':
    sys.exit(build())
