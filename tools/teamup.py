# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Team Up name table for the year-end Team Up select (MGFIN file 0x2a8,
RAM 0x8009c2a8). 21 records {u8 string (1/2 = MBMS13 g1 s0001/s0002),
u8 offset in u16 units, u8 length in u16 units, u8 0}, indexed by Team Up
id - 1. MGFIN copies the chosen name out of the list string with it, so
it must follow the EN MBMS13 (private/re/teamup/findings.org).
build_mbms.py MBMS13 calls write() after hook.py has written MGFIN.BIN.
Usage: teamup.py   (prints the table for private/build/MBMS13.BIZ.lz)"""
import struct
OFF = 0x2a8
JP = 'private/unpacked/MBMS13.BIZ.lz'
DISC = 'private/disc/DAT1/MGFIN.BIN'
BUILD = 'private/build/MGFIN.BIN'

def lines(d, s):
    g1 = struct.unpack_from('<I', d, 0x2d000 + 4)[0] - 0x800ce000
    p = struct.unpack_from('<I', d, g1 + 4 * s)[0] - 0x800ce000
    out, k, start = [], 0, 0
    while True:
        c = struct.unpack_from('<H', d, p + 2 * k)[0]
        if c in (0x3000, 0x3001, 0):
            out.append((start, k - start))
            if c != 0x3000: return out
            start = k + 1
        k += 1

def table(en):
    """New 84-byte table for EN MBMS13 data (decompressed)."""
    jp = open(JP, 'rb').read()
    fin = open(DISC, 'rb').read()
    old = [struct.unpack_from('<4B', fin, OFF + 4 * i) for i in range(21)]
    jl = {s: lines(jp, s) for s in (1, 2)}
    el = {s: lines(en, s) for s in (1, 2)}
    out = bytearray()
    for i, (s, o, n, z) in enumerate(old):
        row = [k for k, (st, ln) in enumerate(jl[s]) if st == o]
        assert len(row) == 1 and jl[s][row[0]][1] == n, (i, s, o, n)
        eo, ln = el[s][row[0]]
        assert eo < 256 and ln < 256, (i, eo, ln)
        out += bytes([s, eo, ln, 0])
    return bytes(out)

def write(en):
    d = bytearray(open(BUILD, 'rb').read())
    d[OFF:OFF + 84] = table(en)
    open(BUILD, 'wb').write(d)

if __name__ == '__main__':
    print(table(open('private/build/MBMS13.BIZ.lz', 'rb').read()).hex())
