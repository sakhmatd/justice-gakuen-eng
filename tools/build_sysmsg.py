# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Write the exe's glyph-coded system messages (memory card, options, key
config, records, extras) in English.

Reads tl/SYSMSG.txt, the column limits in data/sysmsg_use.tsv
and the font codes in data/latin.tsv. Edits private/build/SLPS_012.40
(run after hook.py and build_title.py). --check only reports problems.

Format: u16 len + len u16 codes, code 0xTRC = MOJI2 font tpage T, row R,
col C; 0xfff is a blank cell. Each string is one row of a 24-column grid
(exe 0x80022a24, no bounds check), so a string must fit the columns its
screen gives it. Pointer table: 148 u32 at RAM 0x8008a3b4. New strings go
to the old pool, then to free RAM 0x80099d20-0x8009a000 (inside the exe
image, after the BSS clear), then to the hook padding after the hook.
Row-1 titles are drawn with 40x40 glyphs; TITLE_SMALL switches those calls
to normal 20 px glyphs so Latin titles can be used.
EN '=' or empty keeps the original string.
"""
import sys, os, re, struct
sys.path.insert(0, os.path.dirname(__file__))

EXE = 'private/build/SLPS_012.40'
PTRS, NPTR = 0x8008a3b4, 148
POOLS = [(0x80089b90, 0x8008a3b4), (0x80099d20, 0x8009a000)]
HOOK_END = 0x80010d18
SPARE_FROM = 0x80010c38                 # 0x80010c38-0x80010d18 reserved for code
TITLE_SMALL = [0xd864, 0xdd08, 0xe118, 0xeb50, 0x10bb8]   # addiu a3,zero,1 -> 0
SPECIAL = {' ': 0xfff, '.': 0x2b6, ',': 0x070, '-': 0x071, "'": 0x066}
ICONS = {'{CIRCLE}': 0x2a0, '{SQUARE}': 0x2a1, '{CROSS}': 0x2a2, '{TRIANGLE}': 0x2a3,
         '{L1}': 0x2a4, '{L2}': 0x2a5, '{R1}': 0x2a6, '{R2}': 0x2a7}
def F(a): return a - 0x80010000 + 0x800

def charmap():
    m = {}
    for line in open('data/latin.tsv', encoding='utf-8').read().split('\n')[1:]:
        if '\t' not in line: continue
        name, code = line.split('\t')
        if len(name) == 1: m.setdefault(name, int(code, 16))
    m.update(SPECIAL)
    return m

def load_tl(path='tl/SYSMSG.txt'):
    out = {}; sid = None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('## SYSMSG s'): sid = int(line[11:])
        elif line.startswith('EN: ') and sid is not None: out[sid] = line[4:].strip()
    return out

def limits():
    lim = {}
    for line in open('data/sysmsg_use.tsv', encoding='utf-8').read().split('\n')[1:]:
        f = line.split('\t')
        if len(f) >= 3 and f[2].strip().isdigit(): lim[int(f[0][1:])] = int(f[2])
    return lim

def encode(text, cm):
    codes = []
    for tok in re.split(r'(\{[A-Z0-9]+\}|\{sp\})', text):
        if not tok: continue
        if tok == '{sp}': codes.append(0xfff)
        elif tok in ICONS: codes.append(ICONS[tok])
        else:
            for ch in tok:
                if ch not in cm: raise ValueError(f'no glyph for {ch!r}')
                codes.append(cm[ch])
    return codes

def build(check=False):
    cm = charmap(); tl = load_tl(); lim = limits()
    exe = bytearray(open(EXE, 'rb').read())
    ptr = [struct.unpack_from('<I', exe, F(PTRS) + 4 * i)[0] for i in range(NPTR)]
    from hook import build_hook, HOOK
    a = HOOK + len(build_hook()); a += -a % 4
    # the hook padding after the hook is kept free for code
    # (the fighting-mode name helper); only use it as a last resort
    pools = POOLS + [(a, min(HOOK_END, SPARE_FROM))]
    # original strings, keyed by pointer (several ids share one string)
    orig = {}
    for p in set(ptr):
        n = struct.unpack_from('<H', exe, F(p))[0]
        orig[p] = list(struct.unpack_from('<%dH' % n, exe, F(p) + 2))
    problems = 0; new = {}
    for i in range(NPTR):
        en = tl.get(i, '')
        if en in ('', '='): continue
        try: codes = encode(en, cm)
        except ValueError as e: print(f's{i:03d}: {e}'); problems += 1; continue
        if i in lim and len(codes) > lim[i]:
            print(f's{i:03d}: {len(codes)} cells > {lim[i]}: {en}'); problems += 1
        new[i] = codes
    if check:
        print(f'SYSMSG: {len(new)} translated, problems {problems}'); return problems
    # lay out: every distinct pointer gets one string (EN of its first id)
    blobs = {}
    for i in range(NPTR):
        p = ptr[i]
        if p in blobs: continue
        blobs[p] = new.get(i, orig[p])
    for lo, hi in POOLS: exe[F(lo):F(hi)] = bytes(hi - lo)
    cur = [list(x) for x in pools]; where = {}
    for p, codes in sorted(blobs.items(), key=lambda kv: -len(kv[1])):
        size = 2 + 2 * len(codes); size += -size % 4
        for c in cur:
            if c[0] + size <= c[1]:
                where[p] = c[0]; c[0] += size; break
        else:
            print('out of pool space'); return problems + 1
        struct.pack_into('<H%dH' % len(codes), exe, F(where[p]), len(codes), *codes)
    for i in range(NPTR):
        struct.pack_into('<I', exe, F(PTRS) + 4 * i, where[ptr[i]])
    titles = (42, 62, 63, 64, 90)
    for o in (TITLE_SMALL if all(t in new for t in titles) else []):
        w = struct.unpack_from('<I', exe, o)[0]
        assert w in (0x24070001, 0x24070000), hex(o)
        struct.pack_into('<I', exe, o, 0x24070000)
    open(EXE, 'wb').write(exe)
    used = sum(p[0] - q[0] for p, q in zip(cur, pools))
    print(f'SYSMSG: {len(new)} translated, {used} bytes used, problems {problems}')
    return problems

if __name__ == '__main__':
    build('--check' in sys.argv)
