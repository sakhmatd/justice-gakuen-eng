# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Build the patched disc image.

Every file in private/build/ whose name matches a disc file (e.g. MBMS00.BIZ,
MGAME.BIN) replaces it. The exe's file table (12-byte entries at file
offset 0x7109c: BCD MSF, size, param) is rewritten for the new layout.
Output: private/out/justice-en.bin/.cue
Run from the project root.
"""
import os, re, struct, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
from filetab import bcd, load_lba

WORK = 'private'
XML = os.path.join(WORK, 'disc.xml')
OUT = os.path.join(WORK, 'out')
TAB, NTAB = 0x7109c, 842

def tobcd(n): return (n // 10) << 4 | n % 10

def msf(lba):
    a = lba + 150
    return bytes([tobcd(a // 4500), tobcd(a // 75 % 60), tobcd(a % 75), 0])

def lba_of(entry):
    m, s, f = (bcd(x) for x in entry[:3])
    return (m * 60 + s) * 75 + f - 150

def build_xml(repl, exe_path, path):
    xml = open(XML).read()
    def sub(m):
        name = m.group(1)
        if name in repl:
            return m.group(0).replace('<file ', f'<file source="{os.path.abspath(repl[name])}" ', 1)
        if name == 'SLPS_012.40':
            return m.group(0).replace('<file ', f'<file source="{os.path.abspath(exe_path)}" ', 1)
        return m.group(0)
    xml = re.sub(r'<file name="([^"]+)"[^>]*/>', sub, xml)
    xml = xml.replace('file="disc/', f'file="{os.path.abspath(os.path.join(WORK, "disc"))}/')
    xml = xml.replace('source="disc"', f'source="{os.path.abspath(os.path.join(WORK, "disc"))}"')
    open(path, 'w').write(xml)

def mkiso(xml, name):
    subprocess.run(['mkpsxiso', '-y', '-q', '-o', f'{OUT}/{name}.bin', '-c', f'{OUT}/{name}.cue', xml],
                   check=True)
    with open(f'{WORK}/lba_new.txt', 'w') as o:
        subprocess.run([sys.executable, 'tools/isols.py', f'{OUT}/{name}.bin'], check=True, stdout=o)
    return load_lba(f'{WORK}/lba_new.txt')

def main():
    os.makedirs(OUT, exist_ok=True)
    repl = {f: os.path.join(WORK, 'build', f) for f in os.listdir(os.path.join(WORK, 'build'))
            if not f.endswith('.lz')}
    old = load_lba(f'{WORK}/lba.txt')               # lba -> (size, path)
    exe = bytearray(open(os.path.join(WORK, 'disc', 'SLPS_012.40'), 'rb').read())
    if 'SLPS_012.40' in repl: exe = bytearray(open(repl.pop('SLPS_012.40'), 'rb').read())
    # table index -> disc path, from the original layout
    names = []; keep_size = []
    for k in range(NTAB):
        e = exe[TAB + 12*k: TAB + 12*k + 12]
        size, path = old[lba_of(e)]
        names.append(path)
        # XA/STR entries hold sizes in 2336-byte units; leave those alone
        keep_size.append(struct.unpack_from('<I', e, 4)[0] != size)
    exe_path = os.path.join(WORK, 'out', 'SLPS_012.40')
    xml = os.path.join(WORK, 'out', 'build.xml')
    for rnd in range(2):
        open(exe_path, 'wb').write(exe)
        build_xml(repl, exe_path, xml)
        new = mkiso(xml, 'justice-en')
        bypath = {p: (l, s) for l, (s, p) in new.items()}
        changed = 0
        for k, p in enumerate(names):
            l, s = bypath[p]
            e = msf(l) + (exe[TAB + 12*k + 4: TAB + 12*k + 8] if keep_size[k] else struct.pack('<I', s))
            if exe[TAB + 12*k: TAB + 12*k + 8] != e:
                exe[TAB + 12*k: TAB + 12*k + 8] = e; changed += 1
        print(f'round {rnd}: {changed} table entries updated')
        if changed == 0: break
    else:
        sys.exit('file table did not converge')
    print('wrote', f'{OUT}/justice-en.bin')

if __name__ == '__main__':
    main()
