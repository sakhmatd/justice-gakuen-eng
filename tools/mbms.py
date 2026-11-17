# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""MBMS* message files: 4bpp TIM font (20x20 glyphs) + text at 0x2d000.

Loaded at 0x800ce000. Text area: 6 u32 pointers to groups at 0x2d000, each
group = u16 strings followed by a u32 pointer table to them.
"""
import struct, glob, os, hashlib
LOAD = 0x800ce000
TEXT = 0x2d000
GW = GH = 20; GB = 200; FBASE = 0x180
UNPACKED = os.path.join(os.path.dirname(__file__), '..', 'private', 'unpacked')

def font_count(d):
    blen, x, y, w, h = struct.unpack_from('<IHHHH', d, 0x174)
    return h // GH
def glyph(d, i):
    return d[FBASE + i*GB: FBASE + (i+1)*GB]
def gid(g):
    return hashlib.sha1(g).hexdigest()[:12]
def groups(d):
    """Return [(table_offset, [string offsets])] for each non-null group."""
    tops = [t - LOAD for t in struct.unpack_from('<8I', d, TEXT) if t]
    out = []
    for gi, to in enumerate(tops):
        ptrs = []
        o = to
        while o + 4 <= len(d):
            p = struct.unpack_from('<I', d, o)[0]
            if not (LOAD + TEXT < p < LOAD + to): break
            if ptrs and o >= min(ptrs + [len(d)]) and False: break
            ptrs.append(p - LOAD); o += 4
            # stop at next group's string block
            if gi + 1 < len(tops) and o >= next_block(d, tops[gi+1]): break
        out.append((to, ptrs))
    return out
def next_block(d, to):
    """Offset of the first string of the group whose table is at `to`."""
    return struct.unpack_from('<I', d, to)[0] - LOAD
def files():
    return sorted(glob.glob(os.path.join(UNPACKED, 'MBMS*.BIZ.lz')))
