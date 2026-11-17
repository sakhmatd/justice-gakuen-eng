# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Run the hook in a minimal MIPS interpreter and check the composed glyphs."""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import hook, digraph

def run(mem_base, mem, a1, v0=0x80001234, ra=0x80005678):
    code = hook.build_hook()
    reg = [0]*32; reg[5] = a1; reg[2] = v0; reg[31] = ra
    lo = 0; pc = hook.HOOK; delay = None; steps = 0
    def rd(addr, n):
        o = addr - mem_base; return int.from_bytes(mem[o:o+n], 'little')
    def wr(addr, n, v):
        o = addr - mem_base; mem[o:o+n] = (v & ((1 << 8*n) - 1)).to_bytes(n, 'little')
    while True:
        steps += 1; assert steps < 100000
        w = struct.unpack_from('<I', code, pc - hook.HOOK)[0]
        op, rs, rt, rd_, sh, fn = w >> 26, w >> 21 & 31, w >> 16 & 31, w >> 11 & 31, w >> 6 & 31, w & 63
        imm = w & 0xffff; simm = imm - 0x10000 if imm & 0x8000 else imm
        npc = delay if delay is not None else pc + 4; delay = None
        R = lambda i: reg[i] & 0xffffffff
        def setr(i, v):
            if i: reg[i] = v & 0xffffffff
        btarget = pc + 4 + (simm << 2)
        if op == 0:
            if fn == 0x21: setr(rd_, R(rs) + R(rt))
            elif fn == 0x23: setr(rd_, R(rs) - R(rt))
            elif fn == 0x2b: setr(rd_, int(R(rs) < R(rt)))
            elif fn == 0x00: setr(rd_, R(rt) << sh)
            elif fn == 0x02: setr(rd_, R(rt) >> sh)
            elif fn == 0x1b: lo = R(rs) // R(rt)
            elif fn == 0x12: setr(rd_, lo)
            elif fn == 0x08:
                assert R(rs) == v0; return R(5), R(31)
            else: raise ValueError(hex(w))
        elif op == 1:
            if (R(rs) ^ 0x80000000) >= 0x80000000: delay = btarget   # bgez
        elif op == 4:
            if R(rs) == R(rt): delay = btarget
        elif op == 5:
            if R(rs) != R(rt): delay = btarget
        elif op == 9: setr(rt, R(rs) + simm)
        elif op == 0xc: setr(rt, R(rs) & imm)
        elif op == 0xd: setr(rt, R(rs) | imm)
        elif op == 0xf: setr(rt, imm << 16)
        elif op == 0x24: setr(rt, rd(R(rs) + simm, 1))
        elif op == 0x23: setr(rt, rd(R(rs) + simm, 4))
        elif op == 0x28: wr(R(rs) + simm, 1, R(rt))
        elif op == 0x2b: wr(R(rs) + simm, 4, R(rt))
        else: raise ValueError(hex(w))
        pc = npc

if __name__ == '__main__':
    mem = bytearray(open('private/build/MBMS00.BIZ.lz', 'rb').read())
    L = digraph.letter_font()
    ok = 0
    for k, pair in enumerate(['Hi', 'g!', '~ ', 'Wy']):
        for signed in (False,):
            code = 0x8000 | ord(pair[0]) << 7 | ord(pair[1])
            c = code - 0x10000 if signed else code
            a1 = (hook.FONTPIX + c*200) & 0xffffffff
            src, ra = run(hook.MBMS, mem, a1)
            got = mem[src - hook.MBMS: src - hook.MBMS + 200]
            l, r = [L[(ord(x)-0x20)*100:(ord(x)-0x20)*100+100] for x in pair]
            exp = b''.join(l[i*5:i*5+5] + r[i*5:i*5+5] for i in range(20))
            assert got == exp and ra == 0x80005678, pair
            ok += 1
    # normal glyph passes through unchanged
    a1 = hook.FONTPIX + 209*200
    assert run(hook.MBMS, mem, a1)[0] == a1
    # sources outside the pair range pass through (fonts below or far above)
    for a1 in (hook.FONTPIX - 200, 0x800c0000, hook.FONTPIX + 0x10000*200, 0x80180000):
        assert run(hook.MBMS, mem, a1)[0] == a1, hex(a1)
    st = struct.unpack_from('<I', mem, hook.STATE_OFF)[0]
    print('ok', ok, 'ring index', st)
