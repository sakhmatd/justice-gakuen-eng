# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Runtime letter-pair glyphs.

Text codes 0x8000 | c1<<7 | c2 (c1, c2 ASCII 0x20-0x7e) are drawn as one
20x20 cell with two 10x20 letters. The four text pre-render functions in
MGAME, MGENT, MGFIN and MGTIT compute src = 0x800ce180 + code*200 and call
LoadImage through v0 (jalr v0). Each jalr becomes jal HOOK. The hook passes
normal codes through. For pair codes it copies the two letters from the
letter font (in the MBMS file) into a ring buffer and passes that instead.

HOOK lives in unused padding of the exe (0x80010aa0, 644 free bytes).
Letter font, ring index and ring live in the free font area of each
translated MBMS file (see LF_OFF etc.).

Usage: hook.py   -> writes patched SLPS_012.40, MGAME/MGENT/MGFIN/MGTIT.BIN
                    to private/build/
"""
import struct, os, sys
sys.path.insert(0, os.path.dirname(__file__))

HOOK = 0x80010aa0
HOOK_END = 0x80010d18
MBMS = 0x800ce000
FONTPIX = MBMS + 0x180
LF_OFF = 0x12000                # letter font: 96 x 100 bytes (20 rows x 5 bytes)
STATE_OFF = 0x14600             # ring index (u32)
RING_OFF = 0x14800              # 128 x 200 bytes
RING_N = 128
LF, STATE, RING = MBMS + LF_OFF, MBMS + STATE_OFF, MBMS + RING_OFF

SITES = {
    'MGAME.BIN': (0x8009c000, [0x800a32c4, 0x800a3500, 0x800a369c, 0x800a3794]),
    'MGENT.BIN': (0x80174000, [0x8017f938, 0x8017fb74, 0x8017fd10, 0x8017fe08]),
    'MGFIN.BIN': (0x8009c000, [0x8009edd0, 0x8009f00c, 0x8009f118, 0x8009f2b0]),
    'MGTIT.BIN': (0x8009c000, [0x8009ee68, 0x8009f0a4, 0x8009f240, 0x8009f338]),
}
JALR_V0 = 0x0040f809

R = {n: i for i, n in enumerate('zero at v0 v1 a0 a1 a2 a3 t0 t1 t2 t3 t4 t5 t6 t7 '
                                's0 s1 s2 s3 s4 s5 s6 s7 t8 t9 k0 k1 gp sp fp ra'.split())}

class Asm:
    def __init__(self, org):
        self.org = org; self.code = []; self.labels = {}; self.fix = []
    def pc(self): return self.org + 4 * len(self.code)
    def label(self, n): self.labels[n] = self.pc()
    def emit(self, w): self.code.append(w & 0xffffffff)
    def r(self, op, rs, rt, rd, sh, fn):
        self.emit(op << 26 | R[rs] << 21 | R[rt] << 16 | R[rd] << 11 | sh << 6 | fn)
    def i(self, op, rs, rt, imm): self.emit(op << 26 | R[rs] << 21 | R[rt] << 16 | (imm & 0xffff))
    def nop(self): self.emit(0)
    def lui(self, rt, imm): self.i(0x0f, 'zero', rt, imm)
    def ori(self, rt, rs, imm): self.i(0x0d, rs, rt, imm)
    def addiu(self, rt, rs, imm): self.i(0x09, rs, rt, imm)
    def andi(self, rt, rs, imm): self.i(0x0c, rs, rt, imm)
    def lbu(self, rt, off, rs): self.i(0x24, rs, rt, off)
    def lhu(self, rt, off, rs): self.i(0x25, rs, rt, off)
    def sh(self, rt, off, rs): self.i(0x29, rs, rt, off)
    def xori(self, rt, rs, imm): self.i(0x0e, rs, rt, imm)
    def slti(self, rt, rs, imm): self.i(0x0a, rs, rt, imm)
    def sltiu(self, rt, rs, imm): self.i(0x0b, rs, rt, imm)
    def sra(self, rd, rt, sh): self.r(0, 'zero', rt, rd, sh, 0x03)
    def lw(self, rt, off, rs): self.i(0x23, rs, rt, off)
    def sb(self, rt, off, rs): self.i(0x28, rs, rt, off)
    def sw(self, rt, off, rs): self.i(0x2b, rs, rt, off)
    def addu(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x21)
    def subu(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x23)
    def and_(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x24)
    def or_(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x25)
    def xor(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x26)
    def sltu(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x2b)
    def sll(self, rd, rt, sh): self.r(0, 'zero', rt, rd, sh, 0x00)
    def srl(self, rd, rt, sh): self.r(0, 'zero', rt, rd, sh, 0x02)
    def jr(self, rs): self.r(0, rs, 'zero', 'zero', 0, 0x08)
    def divu(self, rs, rt): self.r(0, rs, rt, 'zero', 0, 0x1b)
    def mflo(self, rd): self.r(0, 'zero', 'zero', rd, 0, 0x12)
    def li32(self, rt, v):
        self.lui(rt, v >> 16); self.ori(rt, rt, v & 0xffff)
    def branch(self, kind, rs, rt, lab):
        op, rtn = {'beq': (4, rt), 'bne': (5, rt), 'bgez': (1, 'at')}[kind]
        if kind == 'bgez': rtn = 'at'                         # REGIMM rt=1 -> bgez
        self.fix.append((len(self.code), lab)); self.i(op, rs, rtn, 0)
    def finish(self):
        for idx, lab in self.fix:
            off = (self.labels[lab] - (self.org + 4*idx + 4)) >> 2
            self.code[idx] = (self.code[idx] & 0xffff0000) | (off & 0xffff)
        return b''.join(struct.pack('<I', w) for w in self.code)

def mul_const(a, dst, src, k, tmp):
    """dst = src * k using shifts (k = 100 or 200). dst != src."""
    bits = [b for b in range(16) if k >> b & 1]
    a.sll(dst, src, bits[0])
    for b in bits[1:]:
        a.sll(tmp, src, b); a.addu(dst, dst, tmp)

def build_hook():
    a = Asm(HOOK)
    a.li32('t0', FONTPIX)
    a.subu('t1', 'a1', 't0')            # t1 = code*200 for MBMS font sources
    a.lui('t2', 0x0064)                 # 0x8000*200
    a.subu('t3', 't1', 't2')            # pair codes: 0 <= t3 < 0x8000*200 (unsigned)
    a.sltu('t3', 't3', 't2')
    a.branch('beq', 't3', 'zero', 'out')
    a.addiu('t3', 'zero', 200)          # delay
    a.divu('t1', 't3')
    a.mflo('t1')                        # code
    a.andi('t2', 't1', 0x7f)            # c2
    a.srl('t1', 't1', 7)
    a.andi('t1', 't1', 0x7f)            # c1
    a.addiu('t1', 't1', -0x20)
    a.addiu('t2', 't2', -0x20)
    mul_const(a, 't4', 't1', 100, 't5'); mul_const(a, 't6', 't2', 100, 't5')
    a.li32('t5', LF)
    a.addu('t1', 't4', 't5')            # left letter
    a.addu('t2', 't6', 't5')            # right letter
    a.li32('t6', STATE)
    a.lw('t7', 0, 't6')
    a.nop()
    a.addiu('t8', 't7', 1)
    a.andi('t8', 't8', RING_N - 1)
    a.sw('t8', 0, 't6')
    a.andi('t7', 't7', RING_N - 1)
    mul_const(a, 't9', 't7', 200, 't8')
    a.li32('t8', RING)
    a.addu('a1', 't8', 't9')            # new source
    a.addu('t9', 'a1', 'zero')          # dst cursor
    a.addiu('t3', 'zero', 20)           # rows
    a.label('row')
    for k in range(5):
        a.lbu('t4', k, 't1'); a.lbu('t5', k, 't2'); a.nop()
        a.sb('t4', k, 't9'); a.sb('t5', 5 + k, 't9')
    a.addiu('t1', 't1', 5)
    a.addiu('t2', 't2', 5)
    a.addiu('t3', 't3', -1)
    a.branch('bne', 't3', 'zero', 'row')
    a.addiu('t9', 't9', 10)             # delay
    a.label('out')
    a.jr('v0')
    a.nop()
    code = a.finish()
    assert HOOK + len(code) <= HOOK_END, hex(HOOK + len(code))
    return code

def up_tables():
    """Stat UP/DOWN sprite x = 88 + STAT[stat] + CAT[cat] (MGAME 0x800bf454).
    The tables hold the JP pixel widths of the stat name (MBMS00 g4 s0043-50)
    and of the message up to its first {sp} (g4 s0029-42). Recompute them
    from the EN text. The message starts a new cell after the stat name, so
    STAT is the name in whole cells. CAT is the drawn width of the text, so
    a message that ends in a half-filled letter pair does not add 10 px.
    The sprite then starts where the last letter's 10 px column ends; its
    resting frame has 8 px of transparent margin on the left."""
    import re, build_mbms
    tl = open('tl/MBMS00.txt', encoding='utf-8').read()
    def en(i):
        m = re.search(r'## MBMS00 g4 s%04d\nJP: .*\nEN: (.*)' % i, tl)
        return m[1].replace('{p}', '').replace('{end}', '')
    def px(s):
        w = build_mbms.cells(s) * 20
        last = build_mbms.SYMRUN.split(re.split(r'\{[^}]*\}', s)[-1])[-1]
        return w - 10 if len(last) % 2 else w
    cat = [px(en(i).split('{sp}')[0]) for i in range(29, 43)]
    stat = [build_mbms.cells(en(i)) * 20 for i in range(43, 51)]
    return struct.pack('<14H', *cat) + struct.pack('<8H', *stat)

UP_TABLES = 0x3604              # MGAME file offset: CAT (14 u16), STAT (8 u16)
UP_ORIG = struct.pack('<22H', 120, 160, 100, 100, 40, 80, 120, 140, 80, 40, 100, 100,
                      120, 120, 60, 60, 40, 40, 80, 80, 60, 60)

# Name entry opens on the alphanumeric page (2) with the cursor on "A"
# (x 0, y 1). MGENT 0x80174d70 is the screen init; 0x801776f0 makes the
# keyboard's first-frame setup call the page routine 0x8017d100 (used by
# the tabs) so max x/y match the page. (RAM, old word, new word)
MGENT_ROMAJI = [
    (0x80174d70, 0x24020001, 0x24020002),   # addiu v0,zero,2
    (0x80174d78, 0xa0201a2b, 0xa0221a2b),   # sb v0,0x1a2b(at): page = 2
    (0x80174d7c, 0x3c01801f, 0x24020001),   # addiu v0,zero,1 (at = 0x801f)
    (0x80174d88, 0xa0201a27, 0xa0221a27),   # sb v0,0x1a27(at): y = 1
    (0x801776f0, 0x3c02801f, 0x0c05f440),   # jal 0x8017d100
    (0x801776f4, 0x90421a2b, 0x02402021),   # move a0,s2
    (0x801776f8, 0x00000000, 0x0805ddd1),   # j 0x80177744
    (0x801776fc, 0xa2c219ab, 0x00000000),   # nop
]

# Default names (used when a field is left blank): u16 shared-glyph strings
# ending in 0x3001 at MGENT file 0x10 (surname), 0x18 (given), 0x20
# (nickname). Written as letter pairs, which only the hooked diary-mode
# renderers draw. The nickname is also copied to the fighting-mode EDIT
# struct (0x801f5a04+0x10) whose renderers are not hooked.
def pair(a, b=' '): return 0x8000 | ord(a) << 7 | ord(b)
MGENT_NAMES = [
    (0x10, (0xde, 0xd2, 0x100), (pair('S', 'e'), pair('i', 'g'), pair('i'))),     # Seigi
    (0x18, (0xef, 0xda, 0xe4), (pair('M', 'a'), pair('k', 'o'), pair('t', 'o'))),  # Makoto
    (0x20, (0x75, 0x60, 0x6a), (pair('M', 'a'), pair('k', 'o'), pair('t', 'o'))),  # Makoto
]

# Name-insert spacing (task I, topic nameinsert). The MGAME page renderer
# (0x800ade14) advances screen x by 20 per cell. Two blocks are rewritten in
# place without their load-delay nops, using the free registers t3-t7:
#  - name loop 0x800ae038-0x800ae110 (s2 = cell pointer, a3 = 0x061e,
#    t0 = 0x3001): a name cell whose second letter is a space (code &
#    0x807f == 0x8020) adds 10 only when it is the last cell (next cell
#    0x3001) or a suffix follows (next cell starts with '-', "-kun").
#    Inner pairs such as "n " in "Gorin High" add 20. 0x800ae114 (the
#    entry from the insert dispatch) is kept;
#  - plain glyph emit 0x800ae12c-0x800ae1b0 (a1 = code): a half pair
#    (c, 0x7f) (code & 0x807f == 0x807f) adds 10. build_mbms pads the last
#    pair of an odd-length part before an insert, colour or face tag with
#    0x7f (a blank letter font cell).
# MGFIN has a byte-identical copy 0x9c3c bytes lower.
NAMEFIX = [(0x800ae038, 55, 0xb3d41376, 'name'), (0x800ae12c, 34, 0xa262cc3a, 'plain')]
NAMEFIX_DELTA = {'MGAME.BIN': 0, 'MGFIN.BIN': -0x9c3c}

def namefix_code(org, kind):
    a = Asm(org)
    def tile_and_xy():
        # sprite tile x/y, tile x += 20 (wrap at 240), sprite screen x/y
        a.lhu('t5', 0x1540, 's0'); a.lhu('t6', 0x1542, 's0')
        if kind == 'plain': a.xori('t4', 't4', 0x807f)
        a.sh('t5', 0x38, 'a2'); a.sh('t6', 0x3a, 'a2')
        a.addiu('t5', 't5', 20)
        a.sll('t7', 't5', 16); a.sra('t7', 't7', 16); a.slti('t7', 't7', 240)
        a.branch('bne', 't7', 'zero', 'nowrap')
        a.sh('t5', 0x1540, 's0')
        a.addiu('t6', 't6', 20); a.sh('t6', 0x1542, 's0'); a.sh('zero', 0x1540, 's0')
        a.label('nowrap')
        a.lhu('t5', 0x1544, 's0'); a.lhu('t6', 0x1546, 's0')
        if kind == 'plain': a.sltiu('t4', 't4', 1)  # 1 = half cell
        a.sh('t5', 0x3c, 'a2'); a.sh('t6', 0x3e, 'a2')
        a.sll('t7', 't4', 3); a.sll('t4', 't4', 1); a.addu('t4', 't4', 't7')
        a.subu('t5', 't5', 't4'); a.addiu('t5', 't5', 20)
        a.sh('t5', 0x1544, 's0')                    # x += 20 or 10
    if kind == 'name':
        a.lhu('v0', 0, 's2'); a.sb('zero', 0x41, 'a2')
        a.label('skip')                             # blank cells 0x061e: x += 20
        a.branch('bne', 'v0', 'a3', 'cell')
        a.lhu('v1', 0x1544, 's0')                   # delay
        a.addiu('s2', 's2', 2); a.addiu('v1', 'v1', 20)
        a.lhu('v0', 0, 's2')
        a.branch('beq', 'zero', 'zero', 'skip')
        a.sh('v1', 0x1544, 's0')                    # delay
        a.label('cell')
        exit_ = 0x800ae124 + (org - 0x800ae038)
        a.i(4, 'v0', 't0', (exit_ - (a.pc() + 4)) >> 2)   # beq v0, t0, exit
        a.lhu('t7', 2, 's2')                        # delay: next cell
        a.andi('t4', 'v0', 0x807f); a.xori('t4', 't4', 0x8020)
        a.xor('t5', 't7', 't0')                     # 0 = last cell
        a.andi('t7', 't7', 0xff80); a.xori('t7', 't7', 0x9680)  # 0 = "-" follows
        a.sltiu('t5', 't5', 1); a.sltiu('t7', 't7', 1); a.or_('t5', 't5', 't7')
        a.sltiu('t4', 't4', 1); a.and_('t4', 't4', 't5')        # 1 = half cell
        tile_and_xy()
        a.lhu('a0', 0x1548, 's0'); a.addiu('s2', 's2', 2)
        a.addiu('a0', 'a0', 1); a.sh('a0', 0x1548, 's0')
        a.sll('v1', 'a0', 16); a.sra('v1', 'v1', 16)
        a.sll('v0', 'v1', 4); a.addu('v0', 'v0', 'v1'); a.sll('v0', 'v0', 2)
        a.addu('a2', 's0', 'v0'); a.sb('zero', 0x40, 'a2')
    else:
        a.andi('t4', 'a1', 0x807f)
        tile_and_xy()
        a.lbu('v1', 0x1560, 's0'); a.lhu('v0', 0x1548, 's0')
        a.sb('v1', 0x41, 'a2'); a.addiu('v0', 'v0', 1); a.sh('v0', 0x1548, 's0')
    return a.finish()

def namefix(name, d, base=0x8009c000):
    import zlib
    for ram, n, crc, kind in NAMEFIX:
        o = ram + NAMEFIX_DELTA[name] - base
        assert zlib.crc32(d[o:o + 4 * n]) == crc, (name, hex(ram), 'name loop moved')
        code = namefix_code(ram + NAMEFIX_DELTA[name], kind)
        assert len(code) <= 4 * n, (kind, len(code) // 4)
        d[o:o + 4 * n] = code + bytes(4 * n - len(code))

# MGAME menu/status box renderer 0x800b86a8 (hub help texts in MBMS group
# 4 such as Training, and the hub date box). Its glyph loop
# 0x800b8aec-0x800b8cbc adds 20 to x per glyph (0x800b8c74), so a half
# pair (c, 0x7f) left a 10 px gap. s5 counted the line width for the
# newline rewind (x -= s5); the newline now reloads the box start x
# 0xc84(s4), which frees s5. s5 = 0 for a half pair, 1 otherwise, and a
# glyph advances 10 << s5. s5 is callee-saved, so it survives the GPU
# calls in between. MGFIN has no copy (private/re/trainhelp/findings.org).
HELPFIX = [
    (0x800b8af8, 0x97a20010, lambda a: a.lhu('v0', 0xc84, 's4')),   # was lhu v0,0x10(sp)
    (0x800b8b00, 0x00551023, lambda a: a.nop()),                    # was subu v0,v0,s5
    (0x800b8b3c, 0x26b50014, lambda a: a.andi('s5', 'v1', 0x807f)), # was addiu s5,s5,20
    (0x800b8b6c, 0x00000000, lambda a: a.xori('s5', 's5', 0x807f)), # load-delay nops
    (0x800b8b90, 0x00000000, lambda a: a.sltu('s5', 'zero', 's5')),
    (0x800b8c58, 0x00000000, lambda a: a.addiu('t1', 'zero', 10)),
    (0x800b8c70, 0x00000000, lambda a: a.r(0, 's5', 't1', 't1', 0, 0x04)),  # sllv t1,t1,s5
    (0x800b8c74, 0x24420014, lambda a: a.addu('v0', 'v0', 't1')),   # was addiu v0,v0,20
]

# Dates in the hub box and the save lists (private/re/hubdate/findings.org).
# build_mbms makes each number in slot 4 s0019 (0x061e, letter cell) plus
# one trailing 0x061e. The day lookup reads 1 u16 later, so the day is
# (cell, 0x061e), left aligned: replace addiu a1,a1,-1 ; sll a1,a1,2 with
# sll a1,a1,2 ; addiu a1,a1,-2 in each builder.
DAYFIX = {'MGAME.BIN': [0x800b8a3c, 0x800bea20],      # hub box, Backup list
          'MGTIT.BIN': [0x800a702c],                  # Continue list
          'MGFIN.BIN': [0x800ab6a0]}                  # year-end save list
DAY_OLD, DAY_NEW = (0x24a5ffff, 0x00052880), (0x00052880, 0x24a5fffe)
# Save list glyph loop tail (identical code in the three overlays): a half
# pair advances 10. The sprite row counter 0x20(sp) is replaced by fp,
# which already counts 20 per sprite and is reset with it.
LISTFIX = ({'MGAME.BIN': 0x800bed3c, 'MGTIT.BIN': 0x800a7348, 'MGFIN.BIN': 0x800ab9bc},
           25, 0xb3cb4c9f)

def listfix_code(T):
    a = Asm(T)
    a.lhu('a1', 0x10, 'sp'); a.lhu('v1', -2, 's1')     # x, code
    a.sh('a1', 0x38, 's5'); a.lhu('v0', 0x12, 'sp')
    a.addiu('fp', 'fp', 20); a.sh('v0', 0x3a, 's5')
    a.andi('v1', 'v1', 0x807f); a.xori('v1', 'v1', 0x807f)   # 0 = half pair
    a.branch('bne', 'v1', 'zero', 'full')
    a.addiu('a1', 'a1', 20)                             # delay
    a.addiu('a1', 'a1', -10)
    a.label('full')
    a.sh('a1', 0x10, 'sp')
    a.slti('v0', 'fp', 240)                             # 12 sprites per row
    a.branch('bne', 'v0', 'zero', 'join')
    a.addiu('s0', 's0', 1)                              # delay
    a.lhu('t0', 0x18, 'sp'); a.r(0, 'zero', 'zero', 'fp', 0, 0x21)   # move fp,zero
    a.addiu('t0', 't0', 20); a.sh('t0', 0x18, 'sp')
    a.label('join')
    a.addiu('t0', 's5', 0x3c); a.slti('v0', 's0', 12)
    a.i(4, 'v0', 'zero', (T + 0x78 - (a.pc() + 4)) >> 2)  # beqz v0, exit
    a.sw('t0', 0x38, 'sp')                              # delay
    a.nop(); a.nop()                                    # -> T+0x64
    return a.finish()

def textfix(name, d, base=0x8009c000):
    import zlib
    if name == 'MGAME.BIN':
        for ram, old, f in HELPFIX:
            o = ram - base
            assert struct.unpack_from('<I', d, o)[0] == old, (hex(ram), 'menu box loop moved')
            a = Asm(ram); f(a); d[o:o + 4] = a.finish()
    for ram in DAYFIX.get(name, []):
        o = ram - base
        assert struct.unpack_from('<2I', d, o) == DAY_OLD, (name, hex(ram), 'date builder moved')
        struct.pack_into('<2I', d, o, *DAY_NEW)
    sites, n, crc = LISTFIX
    if name in sites:
        o = sites[name] - base
        assert zlib.crc32(d[o:o + 4 * n]) == crc, (name, 'save list loop moved')
        code = listfix_code(sites[name])
        assert len(code) == 4 * n, len(code) // 4
        d[o:o + 4 * n] = code

# MGENT (group 6 text) draws a page as strips: one sprite per run of
# glyphs with the same colour, cut every 12 texture cells. The emit loop
# 0x8017ee0c adds 20 per glyph to screen x (0x10(sp)), line offset a0 and
# strip width a1, and 20 to texture u (t2). The block 0x8017eeb4-0x8017ef90
# (colour code 0x3003, glyph step, full texture row) is rewritten without
# its nops so that a half pair (c, 0x7f) adds 10 to x, a0 and a1 (u still
# 20). It trims the right half of that cell, which is blank. This is only
# right when the half pair ends a strip: in group 6 build_mbms emits it
# only right before a colour tag. t6/t7 are free in this function.
MGENTFIX = (0x8017eeb4, 56, 0x99c1e548)

def mgentfix_code(org):
    a = Asm(org)
    def j(target):                                  # j to an absolute address
        a.emit(0x02 << 26 | (target >> 2 & 0x3ffffff))
    LOOP, END = 0x8017ee0c, 0x8017ef94
    def beqz_abs(rs, target):
        a.i(4, rs, 'zero', (target - (a.pc() + 4)) >> 2)
    a.addiu('v0', 'zero', 0x3003)
    a.branch('bne', 'v1', 'v0', 'glyph')
    a.andi('t6', 'v1', 0x807f)                      # delay
    a.branch('beq', 'a2', 'zero', 'setcol')
    a.lhu('t3', 0, 't0')                            # delay: colour argument
    a.addiu('a3', 'a3', 0x44); a.addiu('s1', 's1', 1)
    a.slti('v0', 's1', 0x14); a.sh('a1', 0x3c, 's0')
    beqz_abs('v0', END)
    a.sh('t5', 0x3e, 's0')
    a.addu('s0', 's2', 'a3'); a.addu('a1', 'zero', 'zero'); a.addu('a2', 'zero', 'zero')
    a.sb('t2', 0x40, 's0'); a.sb('t1', 0x41, 's0')
    a.label('setcol')
    a.lhu('v0', 0x10, 'sp'); a.sh('t3', 0x42, 's0'); a.sh('v0', 0x38, 's0')
    a.lhu('v0', 0x12, 'sp'); a.addiu('t0', 't0', 2)
    j(LOOP); a.sh('v0', 0x3a, 's0')
    a.label('glyph')
    a.xori('t6', 't6', 0x807f)
    a.sltiu('t6', 't6', 1)                          # 1 = half pair (c, 0x7f)
    a.r(0, 't6', 't5', 't7', 0, 0x06)               # srlv t7, t5(=20), t6
    a.lhu('v0', 0x10, 'sp'); a.addiu('t2', 't2', 20)
    a.addu('a0', 'a0', 't7'); a.addu('a1', 'a1', 't7')
    a.addu('v0', 'v0', 't7'); a.sh('v0', 0x10, 'sp')
    a.addiu('t4', 't4', 1); a.slti('v0', 't4', 12)
    a.i(5, 'v0', 'zero', (LOOP - (a.pc() + 4)) >> 2)  # bnez v0, LOOP
    a.addiu('a2', 'a2', 1)
    # full texture row: close the strip, start one on the next row
    a.addiu('a3', 'a3', 0x44); a.addiu('s1', 's1', 1)
    a.slti('v0', 's1', 0x14); a.sh('a1', 0x3c, 's0')
    beqz_abs('v0', END)
    a.sh('t5', 0x3e, 's0')
    a.addu('s0', 's2', 'a3'); a.addu('t2', 'zero', 'zero'); a.addiu('t1', 't1', 20)
    a.addu('a1', 'zero', 'zero'); a.sb('zero', 0x40, 's0'); a.sb('t1', 0x41, 's0')
    a.sh('t3', 0x42, 's0'); a.lhu('v0', 0x10, 'sp')
    a.addu('a2', 'zero', 'zero'); a.sh('v0', 0x38, 's0')
    a.lhu('v0', 0x12, 'sp'); a.addu('t4', 'zero', 'zero')
    j(LOOP); a.sh('v0', 0x3a, 's0')
    return a.finish()

def mgentfix(d, base=0x80174000):
    import zlib
    ram, n, crc = MGENTFIX
    o = ram - base
    assert zlib.crc32(d[o:o + 4 * n]) == crc, 'MGENT emit loop moved'
    code = mgentfix_code(ram)
    assert len(code) <= 4 * n, len(code) // 4
    d[o:o + 4 * n] = code + bytes(4 * n - len(code))

STATUS_ROWS = [(0x20a6, '<H', 336, 356), (0x2061, '<B', 80, 100),
               (0x20ae, '<H', 376, 436), (0x2069, '<B', 120, 180)]
TEAMUP_BOX = [(0x6cc0, 0x00031880, 0x00031840), (0x6cc4, 0x24630007, 0x24630008),
              (0x6ab4, 0x24020094, 0x240200a8)]
CHOICE_SAVELOAD = {'MGAME.BIN': 0x2aea, 'MGFIN.BIN': 0x1706, 'MGTIT.BIN': 0x11c2}

def patch_all(disc='private/disc', out='private/build'):
    os.makedirs(out, exist_ok=True)
    code = build_hook()
    exe = bytearray(open(os.path.join(disc, 'SLPS_012.40'), 'rb').read())
    off = 0x800 + HOOK - 0x80010000
    assert not any(exe[off:off + len(code)]), 'hook area not empty'
    exe[off:off + len(code)] = code
    import nick
    nick.patch_exe(exe)
    open(os.path.join(out, 'SLPS_012.40'), 'wb').write(exe)
    jal = 0x0c000000 | (HOOK >> 2 & 0x3ffffff)
    for name, (base, sites) in SITES.items():
        d = bytearray(open(os.path.join(disc, 'DAT1', name), 'rb').read())
        for s in sites:
            o = s - base
            assert struct.unpack_from('<I', d, o)[0] == JALR_V0, (name, hex(s))
            struct.pack_into('<I', d, o, jal)
        if name in ('MGAME.BIN', 'MGFIN.BIN'):
            import nick
            nick.patch(name, d)
            namefix(name, d)
        if name in ('MGAME.BIN', 'MGFIN.BIN', 'MGTIT.BIN'):
            textfix(name, d)                # help box and date half pairs
        if name == 'MGENT.BIN':
            for ram, old, new in MGENT_ROMAJI:
                o = ram - base
                assert struct.unpack_from('<I', d, o)[0] == old, hex(ram)
                struct.pack_into('<I', d, o, new)
            for o, old, new in MGENT_NAMES:
                assert struct.unpack_from('<3H', d, o) == old, hex(o)
                struct.pack_into('<3H', d, o, *new)
            mgentfix(d)                     # pair (c, 0x7f) = half cell
            import namepack                 # typed Latin letters packed in pairs
            d[:] = namepack.apply(d)
        if name == 'MGFIN.BIN':
            # Team Up select cursor box: 168 px left, 208 px right
            # (width = (col*k+c)*20+8; private/re/teamup/findings.org)
            for o, a, b in TEAMUP_BOX:
                assert struct.unpack_from('<I', d, o)[0] == a, hex(o)
                struct.pack_into('<I', d, o, b)
        if name in CHOICE_SAVELOAD:
            # Save/Load menu: one choice number per glyph sprite, laid out
            # for JP 3+3 glyphs; EN "Save{n}Load" has 2+2 pair cells
            # (private/re/loadcol/findings.org)
            o = CHOICE_SAVELOAD[name]
            assert d[o:o + 6] == bytes([0, 0, 0, 1, 1, 1]), name
            d[o:o + 6] = bytes([0, 0, 1, 1, 1, 1])
        if name == 'MGAME.BIN':
            # Status screen: EN names (s0001) need 29 glyphs, the VRAM area
            # holds 24; give s0001 a third row by moving objects 2 and 3
            # down (private/re/status/findings.org)
            for o, fmt, old, new in STATUS_ROWS:
                assert struct.unpack_from(fmt, d, o)[0] == old, hex(o)
                struct.pack_into(fmt, d, o, new)
            assert d[UP_TABLES:UP_TABLES + 44] == UP_ORIG, 'UP tables moved'
            d[UP_TABLES:UP_TABLES + 44] = up_tables()
        open(os.path.join(out, name), 'wb').write(d)
    print(f'hook {len(code)} bytes at {HOOK:#x}; patched', ', '.join(SITES))

if __name__ == '__main__':
    patch_all()
