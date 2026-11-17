# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Make the release patches from the original dump and the built image.

Usage: python3 tools/make_release.py VERSION [PATCHED.bin]

PATCHED.bin defaults to private/out/justice-en.bin. The original is the dump
named in tools/patch_bin.py (ORIG). Writes private/dist/justice-en-VERSION/ with
a .bps and a .ppf patch, apply_patch.py and README.txt, and zips it to
private/dist/justice-en-VERSION.zip, with LICENSE and LICENSE-TEXT. Both patches are applied to a temporary copy
and checked against PATCHED.bin before the zip is written.
Run from the project root. Needs numpy.
"""
import hashlib, os, shutil, struct, sys, zipfile, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from patch_bin import ORIG
import apply_patch

CHUNK = 1 << 26


def diff_runs(a_path, b_path, gap):
    """(start, end) byte ranges where the files differ; gaps < gap merge."""
    idx = []
    with open(a_path, 'rb') as a, open(b_path, 'rb') as b:
        base = 0
        while True:
            x, y = a.read(CHUNK), b.read(CHUNK)
            if not x:
                break
            d = np.nonzero(np.frombuffer(x, np.uint8) != np.frombuffer(y, np.uint8))[0]
            idx.append(d + base)
            base += len(x)
    idx = np.concatenate(idx)
    if not len(idx):
        return []
    br = np.nonzero(np.diff(idx) > gap)[0]
    starts = idx[np.r_[0, br + 1]]
    ends = idx[np.r_[br, len(idx) - 1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def varint(n):
    out = bytearray()
    while True:
        x = n & 0x7f
        n >>= 7
        if n == 0:
            out.append(0x80 | x)
            return bytes(out)
        out.append(x)
        n -= 1


def write_bps(src, dst, path):
    size = os.path.getsize(src)
    assert os.path.getsize(dst) == size, 'the patcher keeps the image size'
    p = bytearray(b'BPS1' + varint(size) + varint(size) + varint(0))
    pos = 0
    with open(dst, 'rb') as t:
        for s, e in diff_runs(src, dst, 8):
            if s > pos:
                p += varint((s - pos - 1) << 2 | 0)          # SourceRead
            t.seek(s)
            p += varint((e - s - 1) << 2 | 1) + t.read(e - s)  # TargetRead
            pos = e
    if pos < size:
        p += varint((size - pos - 1) << 2 | 0)
    p += struct.pack('<II', apply_patch.crc32_file(src), apply_patch.crc32_file(dst))
    p += struct.pack('<I', zlib.crc32(p) & 0xffffffff)
    open(path, 'wb').write(p)


def write_ppf(src, dst, path, desc):
    p = bytearray(b'PPF30\x02' + desc.encode('ascii')[:50].ljust(50, b' '))
    p += bytes([0, 1, 0, 0])                 # BIN image, block check, no undo
    with open(src, 'rb') as f:
        f.seek(0x9320)
        p += f.read(1024)
    with open(dst, 'rb') as t:
        for s, e in diff_runs(src, dst, 9):
            t.seek(s)
            data = t.read(e - s)
            for k in range(0, len(data), 255):
                c = data[k:k + 255]
                p += struct.pack('<QB', s + k, len(c)) + c
    open(path, 'wb').write(p)


README = """\
Shiritsu Justice Gakuen - Legion of Heroes: English patch {ver}
=================================================================

This patch translates the diary mode of the Japanese Evolution Disc
(Disc 1, SLPS-01240) into English. You need your own dump of that disc.
The patch does not contain the game.

By Sergei Akhmatdinov (sakhmatd). Copyright 2026.
apply_patch.py: Apache License 2.0 (LICENSE).
Translation text: Creative Commons BY-NC-SA 4.0 (LICENSE-TEXT).
The game and its Japanese text belong to Capcom. This is an unofficial
fan project, not affiliated with Capcom.

The patch expects the Redump dump (one MODE2/2352 track):
  file size {size} bytes
  CRC32     {crc:08x}
  MD5       {md5}
  SHA-1     {sha}

Patched image:
  CRC32     {pcrc:08x}
  SHA-1     {psha}

Applying the patch
------------------
Use either file:
  {bps}  for Flips, Rom Patcher JS (in a browser) or apply_patch.py
  {ppf}  for PPF-O-Matic, apply_patch.py, or emulators that load PPF

With Python 3 (any system):
  python3 apply_patch.py {bps} "ORIGINAL.bin"

This writes "Shiritsu Justice Gakuen (English).bin" and a .cue next to
the original and leaves the original unchanged. Open the .cue in your
emulator or burn it.

If the check fails, the dump differs from the Redump one (another dump,
or a file converted to .ecm, .chd or .iso). Make a raw 2352-byte
dump, or convert yours back to .bin/.cue, and try again.
"""


def digest(path, algo='sha1'):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__.strip())
        return 2
    ver = sys.argv[1]
    patched = sys.argv[2] if len(sys.argv) == 3 else 'private/out/justice-en.bin'
    name = f'justice-en-{ver}'
    out = os.path.join('private', 'dist', name)
    os.makedirs(out, exist_ok=True)
    bps, ppf = f'{name}.bps', f'{name}.ppf'
    print('BPS...')
    write_bps(ORIG, patched, os.path.join(out, bps))
    print('PPF...')
    write_ppf(ORIG, patched, os.path.join(out, ppf), f'Shiritsu Justice Gakuen English {ver}')
    want = apply_patch.crc32_file(patched)
    tmp = os.path.join('private', 'dist', '.verify.bin')
    try:
        for pf, fn in ((bps, apply_patch.apply_bps), (ppf, apply_patch.apply_ppf)):
            fn(open(os.path.join(out, pf), 'rb').read(), ORIG, tmp)
            got = apply_patch.crc32_file(tmp)
            assert got == want, f'{pf}: CRC32 {got:08x}, expected {want:08x}'
            print(pf, 'verified')
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    shutil.copy(os.path.join(os.path.dirname(__file__), 'apply_patch.py'), out)
    for lic in ('LICENSE', 'LICENSE-TEXT'):
        shutil.copy(lic, out)
    open(os.path.join(out, 'README.txt'), 'w', newline='\r\n').write(README.format(
        ver=ver, size=os.path.getsize(ORIG), crc=apply_patch.crc32_file(ORIG),
        sha=digest(ORIG), md5=digest(ORIG, 'md5'), pcrc=want, psha=digest(patched), bps=bps, ppf=ppf))
    zpath = out + '.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for fn in sorted(os.listdir(out)):
            z.write(os.path.join(out, fn), f'{name}/{fn}')
    for fn in sorted(os.listdir(out)):
        print(f'{os.path.getsize(os.path.join(out, fn)):>10}  {fn}')
    print(f'{os.path.getsize(zpath):>10}  {zpath}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
