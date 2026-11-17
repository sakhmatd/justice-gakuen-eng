# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Replace the Japanese "ただいまロード中..." strip on the random loading screens.

ALOAD32-63 (DAT2) are the loading pictures; each holds a 4bpp 256x32 strip
TIM at 0x3c800 (6 CLUTs at VRAM 512,481; pixels to 0x3d8e0). Only the files
in JP below have the Japanese strip; the builder copies the English
"NOW LOADING" strip TIM from ALOAD33 into them. Writes private/build/ALOADxx.BIZ.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import lz

JP = ['ALOAD34', 'ALOAD38', 'ALOAD3E', 'ALOAD42', 'ALOAD4B', 'ALOAD55', 'ALOAD5C', 'ALOAD5F']
SRC = 'ALOAD33'
A, B = 0x3c800, 0x3d8e0

def build():
    en = open(f'private/unpacked/{SRC}.BIZ.lz', 'rb').read()[A:B]
    problems = 0
    for n in JP:
        d = bytearray(open(f'private/unpacked/{n}.BIZ.lz', 'rb').read())
        assert d[A:A + 8] == en[:8], n
        d[A:B] = en
        c = lz.compress(bytes(d))
        slot = os.path.getsize(f'private/disc/DAT2/{n}.BIZ'); slot += -slot % 2048
        if len(c) > slot: print(f'{n}: {len(c)} > {slot}'); problems += 1; continue
        open(f'private/build/{n}.BIZ.lz', 'wb').write(d)
        open(f'private/build/{n}.BIZ', 'wb').write(c)
    print(f'ALOAD: {len(JP)} strips replaced, problems {problems}')

if __name__ == '__main__':
    build()
