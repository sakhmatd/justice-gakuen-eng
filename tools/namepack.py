# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Letter-pair packing on the name entry screen (MGENT, load 0x80174000).

Rewrites two MGENT functions in place (same entry points, same callers):
  0x8017cd20 input  (called from 0x80177f98, CIRCLE/SQUARE on a key)
  0x8017cebc delete (called from 0x80177a30 CROSS and 0x80177ca4 side button)
Model: the field cursor 0x801f1a2d stays on a cell that holds a single letter.
  Latin key, cell = pair(c1,' ')  -> cell = pair(c1,c), cursor advances
  Latin key, other cell           -> cell = pair(c,' '), cursor stays
  other key                       -> original (store glyph, advance); if the
                                     cell holds a single letter, store into
                                     the next cell (if any) instead
  delete, cell = pair(c1,c2!=' ') -> cell = pair(c1,' '), cursor stays
  delete, other cell              -> original (cell = 0xd0, cursor - 1)
Latin = shared glyphs 1-62 ('1'-'9','0','A'-'Z','a'-'z').

Applied by hook.py to private/build/MGENT.BIN. Usage: namepack.py prints the
listing. Origin and tests: private/re/pack/.
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
import hook
from hook import R

BASE = 0x80174000
INPUT, INPUT_END = 0x8017cd20, 0x8017cebc
DELETE, DELETE_END = 0x8017cebc, 0x8017cf88
SOUND = 0x80180468
FIELDS = 0x80174084              # u32[3] buffer pointers (surname, given, nick)
PAGEGRP = 0x801741bc             # u8 per keyboard page: group 6 string index
G = 0x801f0000                   # 0x1a24 maxx, 0x1a26 x, 0x1a27 y, 0x1a28 focus,
                                 # 0x1a2b page, 0x1a2c side item, 0x1a2d cursor,
                                 # 0x1a2e field, 0x1806 blink timer

class Asm(hook.Asm):
    def lb(self, rt, off, rs): self.i(0x20, rs, rt, off)
    def lhu(self, rt, off, rs): self.i(0x25, rs, rt, off)
    def sh(self, rt, off, rs): self.i(0x29, rs, rt, off)
    def sltiu(self, rt, rs, imm): self.i(0x0b, rs, rt, imm)
    def slt(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x2a)
    def mult(self, rs, rt): self.r(0, rs, rt, 'zero', 0, 0x18)
    def jal(self, t): self.emit(0x0c000000 | (t >> 2 & 0x3ffffff))
    def b(self, lab): self.branch('beq', 'zero', 'zero', lab)
    def xor(self, rd, rs, rt): self.r(0, rs, rt, rd, 0, 0x26)
    def sra(self, rd, rt, sh): self.r(0, 'zero', rt, rd, sh, 0x03)

def field(a, ret):
    """s1 = buffer of field 0x1a2e, s0 = its cell limit (6, nickname 4)."""
    a.lui('t0', G >> 16)
    a.lbu('t1', 0x1a2e, 't0')
    a.lui('t2', FIELDS >> 16)
    a.sltiu('t3', 't1', 3)
    a.branch('beq', 't3', 'zero', ret)
    a.sll('t3', 't1', 2)                    # delay
    a.addu('t2', 't2', 't3')
    a.lw('s1', FIELDS & 0xffff, 't2')
    a.addiu('t2', 'zero', 2)
    a.branch('bne', 't1', 't2', 'f6')
    a.addiu('s0', 'zero', 6)                # delay
    a.addiu('s0', 'zero', 4)
    a.label('f6')

def build_input():
    a = Asm(INPUT)
    a.addiu('sp', 'sp', -0x20)
    a.sw('ra', 0x1c, 'sp'); a.sw('s1', 0x14, 'sp'); a.sw('s0', 0x10, 'sp')
    field(a, 'ret')
    a.lb('t4', 0x1a2d, 't0')
    a.nop()
    a.slt('t4', 't4', 's0')
    a.branch('beq', 't4', 'zero', 'ret')    # cursor past the end: ignore key
    a.nop()
    a.jal(SOUND)
    a.addiu('a0', 'zero', 0x91)
    # glyph = group6[PAGEGRP[page]][y * (maxx + 1) + x]  (as the original)
    a.lui('t0', G >> 16)
    a.lb('v0', 0x1a24, 't0')
    a.lb('a0', 0x1a27, 't0')
    a.addiu('v0', 'v0', 1)
    a.mult('a0', 'v0')
    a.addiu('v1', 'zero', 8)
    a.sh('v1', 0x1806, 't0')
    a.lbu('v1', 0x1a2b, 't0')
    a.lui('a1', 0x8010)
    a.lw('a1', -0x4fe8, 'a1')               # 0x800fb018: group 6 table
    a.lui('at', PAGEGRP >> 16)
    a.addu('at', 'at', 'v1')
    a.lbu('v1', PAGEGRP & 0xffff, 'at')
    a.lb('v0', 0x1a26, 't0')
    a.sll('v1', 'v1', 2)
    a.addu('v1', 'v1', 'a1')
    a.lw('v1', 0, 'v1')
    a.mflo('a2')
    a.addu('a0', 'a0', 'a2')
    a.addu('v0', 'v0', 'a0')
    a.sll('v0', 'v0', 1)
    a.addu('v0', 'v0', 'v1')
    a.lb('t9', 0x1a2d, 't0')                # t9 = cursor p
    a.lhu('v0', 0, 'v0')                    # v0 = glyph
    a.sll('t4', 't9', 1)
    a.addu('t4', 't4', 's1')                # t4 = &cell
    a.lhu('t5', 0, 't4')                    # t5 = cell
    a.ori('t6', 'zero', 0x8020)
    a.andi('t7', 't5', 0x807f)
    a.xor('t7', 't7', 't6')                 # t7 = 0 if cell = pair(c1,' ')
    # shared glyph -> ASCII in t3
    a.addiu('t1', 'v0', -1)
    a.sltiu('t2', 't1', 62)
    a.branch('beq', 't2', 'zero', 'plain')
    a.addiu('t3', 'v0', 60)                 # delay: 37-62 -> 'a'-'z'
    a.sltiu('t2', 'v0', 37)
    a.branch('beq', 't2', 'zero', 'have')
    a.sltiu('t2', 'v0', 11)                 # delay
    a.branch('beq', 't2', 'zero', 'have')
    a.addiu('t3', 'v0', 54)                 # delay: 11-36 -> 'A'-'Z'
    a.addiu('t3', 'v0', 48)                 # 1-9 -> '1'-'9'
    a.addiu('t2', 'zero', 58)
    a.branch('bne', 't3', 't2', 'have')
    a.nop()
    a.addiu('t3', 'zero', 48)               # 10 -> '0'
    a.label('have')
    a.branch('bne', 't7', 'zero', 'single')
    a.addu('t5', 't5', 't3')                # delay
    a.addiu('t5', 't5', -0x20)              # pair(c1, c), advance
    a.b('adv')
    a.sh('t5', 0, 't4')                     # delay
    a.label('single')
    a.sll('t5', 't3', 7)
    a.ori('t5', 't5', 0x8020)               # pair(c, ' '), cursor stays
    a.b('ret')
    a.sh('t5', 0, 't4')                     # delay
    a.label('plain')                        # non-Latin glyph
    a.branch('bne', 't7', 'zero', 'store')
    a.addiu('v1', 't9', 1)                  # delay
    a.slt('t8', 'v1', 's0')                 # cell holds a single letter:
    a.branch('beq', 't8', 'zero', 'store')  # store into the next cell
    a.nop()                                 # unless it is the last one
    a.addu('t9', 'v1', 'zero')
    a.addiu('t4', 't4', 2)
    a.label('store')
    a.sh('v0', 0, 't4')
    a.label('adv')                          # original advance / field-full code
    a.addiu('v0', 't9', 1)
    a.sb('v0', 0x1a2d, 't0')
    a.slt('v0', 'v0', 's0')
    a.branch('bne', 'v0', 'zero', 'ret')
    a.addiu('v0', 's0', -1)                 # delay
    a.sb('v0', 0x1a2d, 't0')                # cursor = last cell
    a.addiu('v0', 'zero', 7)
    a.sb('zero', 0x1a28, 't0')              # focus -> side menu
    a.sb('v0', 0x1a2c, 't0')                # side item 7 (confirm)
    a.sh('zero', 0x1806, 't0')
    a.label('ret')
    a.lw('ra', 0x1c, 'sp'); a.lw('s1', 0x14, 'sp'); a.lw('s0', 0x10, 'sp')
    a.jr('ra')
    a.addiu('sp', 'sp', 0x20)
    code = a.finish()
    assert INPUT + len(code) <= INPUT_END, hex(INPUT + len(code))
    return code + bytes(INPUT_END - INPUT - len(code))

def build_delete():
    a = Asm(DELETE)
    a.addiu('sp', 'sp', -0x18)
    a.sw('ra', 0x14, 'sp'); a.sw('s1', 0x10, 'sp')
    a.jal(SOUND)
    a.addiu('a0', 'zero', 0x92)
    a.sw('s0', 0x0c, 'sp')                  # field() sets s0 too
    field(a, 'ret')
    a.lb('t4', 0x1a2d, 't0')
    a.nop()
    a.sll('t5', 't4', 1)
    a.addu('t5', 't5', 's1')                # &cell
    a.lhu('t6', 0, 't5')
    a.nop()
    a.andi('t7', 't6', 0x8000)
    a.branch('beq', 't7', 'zero', 'erase')
    a.andi('t7', 't6', 0x7f)                # delay
    a.addiu('t8', 'zero', 0x20)
    a.branch('beq', 't7', 't8', 'erase')
    a.andi('t6', 't6', 0xff80)              # delay
    a.ori('t6', 't6', 0x20)                 # pair -> first letter
    a.b('ret')
    a.sh('t6', 0, 't5')                     # delay
    a.label('erase')
    a.addiu('t6', 'zero', 0xd0)
    a.sh('t6', 0, 't5')
    a.addiu('t4', 't4', -1)
    a.branch('bgez', 't4', None, 'st')
    a.nop()
    a.addu('t4', 'zero', 'zero')
    a.label('st')
    a.sb('t4', 0x1a2d, 't0')
    a.label('ret')
    a.lw('ra', 0x14, 'sp'); a.lw('s1', 0x10, 'sp'); a.lw('s0', 0x0c, 'sp')
    a.jr('ra')
    a.addiu('sp', 'sp', 0x18)
    code = a.finish()
    assert DELETE + len(code) <= DELETE_END, hex(DELETE + len(code))
    return code + bytes(DELETE_END - DELETE - len(code))

# Optional cosmetic: screen x of the field cells (u16 tables also used for the
# cursor box at 0x80178ea8). Pitch 24 -> 20 so pairs read as continuous text.
FIELD_X = [
    (0x80174274, (114, 138, 162, 186, 210, 234), (114, 134, 154, 174, 194, 214)),
    (0x80174280, (314, 338, 362, 386, 410, 434), (314, 334, 354, 374, 394, 414)),
    (0x8017428c, (220, 244, 268, 292), (220, 240, 260, 280)),
]

def apply(d, pitch=True):
    d = bytearray(d)
    for ram, old, new in FIELD_X if pitch else []:
        o = ram - BASE; f = '<%dH' % len(old)
        assert struct.unpack_from(f, d, o) == old, hex(ram)
        struct.pack_into(f, d, o, *new)
    for org, code, first in ((INPUT, build_input(), 0x27bdffe0), (DELETE, build_delete(), 0x27bdffe8)):
        assert struct.unpack_from('<I', d, org - BASE)[0] == first, hex(org)
        d[org - BASE:org - BASE + len(code)] = code
    return bytes(d)

if __name__ == '__main__':
    import mips
    for org, code in ((INPUT, build_input()), (DELETE, build_delete())):
        n = len(code.rstrip(b'\0')) // 4 + 1
        print(f'; {org:#x}: {n} words')
        for x, w, t in mips.dis(code, org, org, n): print(f'{x:08x}: {w:08x}  {t}')
