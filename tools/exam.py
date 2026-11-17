# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Exam (SIKEN) text: TESTMJxx.TIZ images (256x256 4bpp, 32x16 cells, 8 per
row) laid out by scripts in DAT1/SIKEN.BIN (load 0x8009c000).

msg scripts:  ptr table at 0x8009c1f8 (30 entries): (cell, n) pairs, 0xfe =
              newline, 0xff = end.
cmd structs:  ptr table at file 0xb94 (30 entries): [ptr order list,
              ptr cmd0, ptr cmd1, ...]; cmd = param, npairs, then (cell, n)
              pairs with 0xff = newline.
"""
import struct
SIKEN = 'private/disc/DAT1/SIKEN.BIN'
B = 0x8009c000
def load(path=SIKEN): return open(path, 'rb').read()
def msg_script(c, lesson):
    p = struct.unpack_from('<I', c, 0x8009c1f8 - B + 4*lesson)[0]
    o = p - B; lines = [[]]
    while True:
        b = c[o]; o += 1
        if b == 0xff: return p, lines, o - (p - B)
        if b == 0xfe: lines.append([]); continue
        lines[-1].append((b, c[o])); o += 1
def cmd_structs(c, lesson):
    sp = struct.unpack_from('<I', c, 0xb94 + 4*lesson)[0]
    nxt = struct.unpack_from('<I', c, 0xb94 + 4*(lesson+1))[0] if lesson < 29 else None
    so = sp - B
    order = struct.unpack_from('<I', c, so)[0]
    cmds = []
    k = 1
    while True:
        q = struct.unpack_from('<I', c, so + 4*k)[0]
        if not (order < q < sp): break
        o = q - B; param = c[o]; n = c[o+1]; o += 2; pairs = []; lines = [[]]
        for _ in range(n):
            if c[o] == 0xff: lines.append([]); o += 1
            lines[-1].append((c[o], c[o+1])); o += 2
        cmds.append((q, param, lines)); k += 1
    return sp, order, cmds
def order_list(c, lesson):
    sp, order, cmds = cmd_structs(c, lesson)
    end = cmds[0][0] if cmds else sp
    return list(c[order - B:end - B])
if __name__ == '__main__':
    c = load()
    for L in range(30):
        p, lines, size = msg_script(c, L)
        sp, order, cmds = cmd_structs(c, L)
        print(f'{L+1:02d} msg {lines}')
        for q, param, cl in cmds: print(f'     cmd@{q:#x} p={param} {cl}')
