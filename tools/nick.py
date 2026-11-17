# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Unpack letter-pair nicknames for the fighting-mode EDIT character.

The diary nickname (0x801f187c, 4 cells + 0x3001) may hold letter pairs
(0x8000 | c1<<7 | c2). The four routines that copy it into the EDIT struct
(0x801f5a04 + 0x10) call NICK instead, which expands pairs into single
shared-glyph Latin codes (1-9, 0 = 10, A-Z 11-36, a-z 37-62, other 0x61e),
keeps 4 glyphs and ends with 0x3001. Unhooked fight renderers (exe
0x8002a2b8, PLSEL1-5, KANRI) then draw "Mako" for "Makoto".
NICK lives in exe padding 0x80010c38-0x80010d10 (see private/re/nick2/).
hook.py applies EXE and the MGAME/MGFIN sites; run this file after
build_exam.py to patch private/build/SIKEN.BIN and write private/build/SKHAY.BIN."""
import os, sys, struct
NICK = 0x80010c38
WORDS = [0x3c01801f, 0x3421187c, 0x24070004, 0x94280000, 0x24210002, 0x3106ffff, 0x30cb007f, 0x30c98000, 0x15200013, 0x30c94000, 0x1120001f, 0x256bff9f, 0x2d69001a, 0x1520001c, 0x25660025, 0x256b0020, 0x2d69001a, 0x15200018, 0x2566000b, 0x256b0010, 0x2d690009, 0x15200014, 0x25660001, 0x256b0001, 0x11600011, 0x2406000a, 0x1000000f, 0x2406061e, 0x000631c2, 0x30c6007f, 0x34c84000, 0x94260000, 0x2569ffe0, 0x15200003, 0x24c6cfff, 0x10c0ffe2, 0x3106ffff, 0x356b4000, 0x000b5c00, 0x010b4025, 0x1000ffdd, 0x3106ffff, 0xa4a60000, 0x24a50002, 0x24e7ffff, 0x10e00005, 0x00084402, 0x1500ffd6, 0x3106ffff, 0x1000ffd2, 0x94280000, 0x24093001, 0x03e00008, 0xa4a90000]
SITES = {
    'MGAME.BIN': [
        (0x65cc, 0x3c02801f, 0x3c05801f),
        (0x65d0, 0x9442187e, 0x34a55a14),
        (0x65d4, 0x3c03801f, 0x8fb20008),
        (0x65d8, 0x94631880, 0x8fb10004),
        (0x65dc, 0x3c04801f, 0x8fb00000),
        (0x65e0, 0x94841882, 0x0800430e),
        (0x65e4, 0x3c05801f, 0x27bd0010),
    ],
    'MGFIN.BIN': [
        (0x24a8, 0x3c02801f, 0x3c05801f),
        (0x24ac, 0x9442187e, 0x34a55a14),
        (0x24b0, 0x3c03801f, 0x8fb20008),
        (0x24b4, 0x94631880, 0x8fb10004),
        (0x24b8, 0x3c04801f, 0x8fb00000),
        (0x24bc, 0x94841882, 0x0800430e),
        (0x24c0, 0x3c05801f, 0x27bd0010),
    ],
    'SIKEN.BIN': [
        (0x1db0, 0x3c04801f, 0x03e07021),
        (0x1db4, 0x9484187c, 0x0c00430e),
        (0x1db8, 0x3c05801f, 0x27050010),
        (0x1dbc, 0x94a5187e, 0x01c0f821),
        (0x1dc0, 0x3c06801f, 0x00000000),
        (0x1dc4, 0x94c61880, 0x00000000),
        (0x1dc8, 0x3c07801f, 0x00000000),
        (0x1dcc, 0x94e71882, 0x00000000),
        (0x1dd0, 0x3c08801f, 0x00000000),
        (0x1dd4, 0x95081884, 0x00000000),
        (0x1de8, 0xa7040010, 0x00000000),
        (0x1dec, 0xa7050012, 0x00000000),
        (0x1df0, 0xa7060014, 0x00000000),
        (0x1df4, 0xa7070016, 0x00000000),
        (0x1df8, 0xa7080018, 0x00000000),
    ],
    'SKHAY.BIN': [
        (0x0438, 0x3c05801f, 0x03e07021),
        (0x043c, 0x94a5187c, 0x0c00430e),
        (0x0440, 0x3c06801f, 0x25e50010),
        (0x0444, 0x94c6187e, 0x01c0f821),
        (0x0448, 0x3c07801f, 0x00000000),
        (0x044c, 0x94e71880, 0x00000000),
        (0x0450, 0x3c08801f, 0x00000000),
        (0x0454, 0x95081882, 0x00000000),
        (0x0458, 0x3c09801f, 0x00000000),
        (0x045c, 0x95291884, 0x00000000),
        (0x0474, 0xa5e50010, 0x00000000),
        (0x0478, 0xa5e60012, 0x00000000),
        (0x047c, 0xa5e70014, 0x00000000),
        (0x0480, 0xa5e80016, 0x00000000),
        (0x0484, 0xa5e90018, 0x00000000),
    ],
}

def patch_exe(exe):
    o = NICK - 0x80010000 + 0x800
    assert not any(exe[o:o + 4 * len(WORDS)]), "NICK area not free"
    struct.pack_into("<%dI" % len(WORDS), exe, o, *WORDS)

def patch(name, d):
    for o, old, new in SITES[name]:
        cur = struct.unpack_from("<I", d, o)[0]
        assert cur in (old, new), (name, hex(o), hex(cur))
        struct.pack_into("<I", d, o, new)

if __name__ == '__main__':
    d = bytearray(open('private/build/SIKEN.BIN', 'rb').read())
    patch('SIKEN.BIN', d); open('private/build/SIKEN.BIN', 'wb').write(d)
    d = bytearray(open('private/disc/DAT1/SKHAY.BIN', 'rb').read())
    patch('SKHAY.BIN', d); open('private/build/SKHAY.BIN', 'wb').write(d)
    print('nick: patched SIKEN.BIN, SKHAY.BIN')
