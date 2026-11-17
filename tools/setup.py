# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Prepare a fresh checkout for building: extract the game data from your
own disc dumps into private/.

Usage: python3 tools/setup.py JP.bin USA.bin

JP.bin   Shiritsu Justice Gakuen - Legion of Heroes (Japan) (Disc 1)
         (Evolution Disc), SLPS-01240, one MODE2/2352 track.
USA.bin  Rival Schools - United by Fate (USA) (Disc 1) (Arcade Disc),
         SLUS-00681. build_fight.py takes the English fight callouts from it.

Writes (all gitignored):
  the dump link at the path tools/patch_bin.py reads (ORIG)
  private/disc/      files of the JP disc (DAT1, DAT2, exe)
  private/lba.txt    JP directory listing
  private/unpacked/  decompressed DAT2 files and archive entries
  private/script/    the dumped MBMS text (for tl_init.py and the tl tools)
  private/usa/disc/, private/usa/unpacked/  the same for the USA disc
Existing outputs are kept; delete them to extract again.
Run from the project root.
"""
import os, subprocess, sys, zlib
sys.path.insert(0, os.path.dirname(__file__))
from lz import decompress
from arc import toc
from patch_bin import ORIG

DISCS = {'JP': (648190032, 0x078c9d65), 'USA': (725629632, 0xbe1ce7d9)}
PY = [sys.executable]


def check(tag, path):
    size, crc = DISCS[tag]
    if os.path.getsize(path) != size:
        sys.exit(f'{tag} dump {path}: {os.path.getsize(path)} bytes, expected {size}')
    c = 0
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            c = zlib.crc32(b, c)
    if c & 0xffffffff != crc:
        sys.exit(f'{tag} dump {path}: CRC32 {c & 0xffffffff:08x}, expected {crc:08x}')
    print(f'{tag} dump OK')


def try_lz(d):
    try:
        out, end = decompress(d)
    except Exception:
        return None
    return out if len(d) - 0x800 <= end <= len(d) and len(out) >= len(d) else None


def unpack(src, dst):
    """Same layout as tools/unpack_all.py."""
    os.makedirs(dst, exist_ok=True)
    for fn in sorted(os.listdir(src)):
        d = open(os.path.join(src, fn), 'rb').read()
        t = toc(d)
        if t:
            os.makedirs(os.path.join(dst, fn), exist_ok=True)
            for i, (o, s) in enumerate(t):
                e = d[o:o + s]
                out = try_lz(e)
                tag = 'lz' if out is not None else 'raw'
                open(os.path.join(dst, fn, f'{i:03d}.{tag}'), 'wb').write(
                    out if out is not None else e)
        else:
            out = try_lz(d)
            tag = 'lz' if out is not None else 'raw'
            open(os.path.join(dst, f'{fn}.{tag}'), 'wb').write(out if out is not None else d)


def step(done, msg, fn):
    if os.path.exists(done):
        print(f'{msg}: {done} exists, skipped')
        return
    print(msg + '...')
    fn()


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip())
    jp, usa = (os.path.abspath(p) for p in sys.argv[1:])
    check('JP', jp)
    check('USA', usa)
    if not os.path.exists(ORIG):
        os.makedirs(os.path.dirname(ORIG), exist_ok=True)
        os.symlink(jp, ORIG)
        print('linked', ORIG)
    tools = os.path.dirname(os.path.abspath(__file__))
    run = lambda *a, **kw: subprocess.run(PY + [os.path.join(tools, a[0])] + list(a[1:]),
                                          check=True, **kw)
    step('private/disc/SLPS_012.40', 'Extracting the JP disc',
         lambda: run('extract.py', jp, 'private/disc'))

    def lba():
        with open('private/lba.txt', 'w') as o:
            run('isols.py', jp, stdout=o)
    step('private/lba.txt', 'Listing the JP disc', lba)
    step('private/unpacked', 'Unpacking JP DAT2', lambda: unpack('private/disc/DAT2', 'private/unpacked'))
    step('private/script', 'Dumping the MBMS text', lambda: run('dump.py'))
    step('private/usa/disc/SLUS_006.81', 'Extracting the USA disc',
         lambda: run('extract.py', usa, 'private/usa/disc'))
    step('private/usa/unpacked', 'Unpacking USA DAT2',
         lambda: unpack('private/usa/disc/DAT2', 'private/usa/unpacked'))
    print('Done. Build with: sh tools/build_all.sh')


if __name__ == '__main__':
    main()
