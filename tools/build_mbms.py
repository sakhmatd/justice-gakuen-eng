# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Rebuild MBMS files with English text in two-letter tiles.

Usage: build_mbms.py [--check] NAME [NAME...]   (e.g. MBMS00)
--check only prints the problems and writes nothing.
Reads translations from tl/*.txt, writes private/build/NAME.BIZ.lz
(decompressed) and private/build/NAME.BIZ (compressed).

Translation entries are keyed by the Japanese segment text. An EN text
without {n} is wrapped automatically per page (40 letters, 4 lines). With
{n} present the layout is taken as written.
"""
import sys, os, re, struct, glob
sys.path.insert(0, os.path.dirname(__file__))
from mbms import *
from dump import kanji_db, charmap, decode
from shared_chars import S
import digraph, lz, hook
LETTERS = digraph.letter_font()

LINE, LINES = 40, 4
MAXG = (TEXT - FBASE) // GB          # glyph slots before the text area
FIRST = 337
NAMEW = 10                           # assumed width of a name insert, letters
TAG = re.compile(r'\{[^}]*\}')
ICON = {v: k for k, v in S.items() if v.startswith('{')}
# Non-ASCII characters that exist in the shared font; each takes a full cell.
SYM = {v: k for k, v in S.items() if len(v) == 1 and not v.isascii() and k > 0}
SYMRUN = re.compile('([' + re.escape(''.join(SYM)) + '])')

def load_tl():
    tl = {}
    for f in sorted(glob.glob('tl/*.txt')):
        jp = None
        for line in open(f, encoding='utf-8'):
            line = line.rstrip('\n')
            if line.startswith('JP: '): jp = line[4:]
            elif line.startswith('EN: ') and jp is not None:
                if line[4:].strip(): tl[jp] = line[4:]
                jp = None
    return tl

def skeleton(t):
    """Control tags that must survive translation, in order."""
    tags = [x for x in TAG.findall(t) if x not in ('{n}', '{sp}') and not x.startswith(('{g:', '{t:', '{x:'))]
    # {p}/{end} order is structural; other tags may move within the segment
    return [x for x in tags if x in ('{p}', '{end}')], sorted(x for x in tags if x not in ('{p}', '{end}'))

CELLS = 20
NAMECELLS = 6                        # a name insert, in cells
# Coded inserts that append a suffix: a 12-letter name + "-kun"/"-san" is 8
# cells, + "-chan" 9 cells (topic nameinsert).
INSCELLS = {5: 8, 6: 8, 7: 8, 8: 9, 9: 9}

def cells(line):
    """Width of one line in 20 px cells, as the encoder lays it out."""
    n = 0
    for part in re.split(r'(\{[^}]*\})', line):
        if not part: continue
        if part.startswith('{'):
            if part == '{sp}' or part.startswith(('{t:', '{g:')) or part in ICON: n += 1
            elif re.match(r'\{v\d', part): n += INSCELLS.get(int(part[2]), NAMECELLS)
        else:
            for q in SYMRUN.split(part):
                if not q: continue
                n += 1 if q in SYM else (len(q) + 1) // 2
    return n

def wrap_page(page):
    """Greedy word wrap of one line of EN text (tags allowed) to CELLS."""
    words = []
    for w in page.split(' '):
        # keep the full name "{v1} {v0}" on one line (private/re/fullname)
        if words and words[-1].endswith('{v1}') and w.startswith('{v0}'):
            words[-1] += ' ' + w
        else: words.append(w)
    lines, cur = [], ''
    for w in words:
        cand = (cur + ' ' + w) if cur else w
        if cur and cells(cand) > CELLS:
            lines.append(cur); cur = w
        else:
            cur = cand
    if cur.strip(): lines.append(cur)
    return lines

def layout(en, maxlines=LINES):
    """Word-wrap every line longer than LINE. Lines that contain {sp} or
    runs of 2+ spaces are table layout and are left alone.
    Returns (text, problems)."""
    probs = []
    out = []
    for part in re.split(r'(\{p\}|\{end\})', en):
        if part in ('{p}', '{end}') or not part: out.append(part); continue
        lines = []
        for line in part.split('{n}'):
            if '{sp}' in line or '  ' in line: lines.append(line)
            else: lines += wrap_page(line) or ['']
        if len(lines) > maxlines: probs.append(f'{len(lines)} lines: {part[:60]}')
        out.append('{n}'.join(lines))
    return ''.join(out), probs

HALF = 0x807f                        # pair (c, 0x7f): c, then advance 10 px

# Date number table (private/re/hubdate/findings.org). Group slot 4 s0019 is
# read as 2 u16 per number by the hub box and the three save lists. Each
# number n becomes (0x061e, one letter cell): (digit, 0x7f) for 1-9, the
# pair "nn" for 10-31, plus a trailing 0x061e for day 31 (the day lookup
# reads 1 u16 later, hook.py DAYFIX). MBMS00 slot 0 s0007 has the same
# JP text (MGENT birthday entry) and keeps its tl EN "=", so this text is
# set by slot here and not in the tl file.
NUMKEY = ('{sp}1{sp}2{sp}3{sp}4{sp}5{sp}6{sp}7{sp}8{sp}9'
          + ''.join(map(str, range(10, 32))) + '{p}{end}')
def date_numbers():
    p = lambda n: (HALF | (0x30 + n) << 7 if n < 10
                   else 0x8000 | (0x30 + n // 10) << 7 | (0x30 + n % 10))
    vals = [v for n in range(1, 32) for v in (0x061e, p(n))] + [0x061e]
    return ''.join('{x:%04x}' % v for v in vals) + '{p}{end}'
SLOT_EN = {(4, NUMKEY): date_numbers()}

class Encoder:
    def __init__(self):
        self.pairs = {}              # '{t:..}' tile -> glyph index
        self.npairs = 0
    def pair(self, p):
        if p == '  ': return 0x061e
        if p.startswith('{t:'):
            if p not in self.pairs: self.pairs[p] = FIRST + len(self.pairs)
            return self.pairs[p]
        a, b = p
        if not (' ' <= a <= '~' and ' ' <= b <= '~'):
            raise ValueError(f'non-ASCII in text: {p!r}')
        self.npairs += 1
        return 0x8000 | ord(a) << 7 | ord(b)
    def encode(self, text):
        vals = []
        for line in re.split(r'(\{n\}|\{p\}|\{end\})', text):
            if not line: continue
            # The last pair of an odd-length text part right before a name
            # insert, a colour tag or a face tag is padded with 0x7f, not
            # ' ' ("{col:0}Diary"{col:-}.", "Yay{col:5}<3", "Hey, {v0}",
            # "Enter your {col:0}name"). The letter font cell 0x7f is
            # blank. MGAME, MGFIN and MGENT advance 10 for (c, 0x7f), so the
            # next part starts right after c.
            table = '{sp}' in line or '  ' in line
            parts = [x for x in re.split(r'(\{[^}]*\})', line) if x]
            for k, part in enumerate(parts):
                if part.startswith('{'):
                    vals += self.tag(part); continue
                q = ''
                for q in SYMRUN.split(part):
                    if not q: continue
                    if q in SYM: vals.append(SYM[q]); continue
                    for i in range(0, len(q), 2):
                        vals.append(self.pair(q[i:i+2].ljust(2)))
                if (not table and k + 1 < len(parts) and re.match(r'\{(v\d|col:|face:)', parts[k + 1])
                        and len(q) % 2 and q not in SYM):
                    vals[-1] = HALF | ord(q[-1]) << 7
        return vals
    def tag(self, t):
        b = t[1:-1]
        if b == 'n': return [0x3000]
        if b == 'p': return [0x3001]
        if b == 'end': return [0]
        if b == 'sp': return [0x061e]
        if t in ICON: return [ICON[t]]
        m = re.fullmatch(r'face:(\d+)', b)
        if m: return [0x3002, int(m[1])]
        m = re.fullmatch(r'col:(-|\d+)', b)
        if m: return [0x3003, 255 if m[1] == '-' else int(m[1])]
        m = re.fullmatch(r'v(\d)(?::(\d+))?', b)
        if m: return [0x2000 + int(m[1])] + ([int(m[2])] if m[2] is not None else [])
        if b.startswith('t:'): return [self.pair(t)]
        m = re.fullmatch(r'g:(\d+)', b)
        if m: return [int(m[1])]
        m = re.fullmatch(r'x:([0-9a-fA-F]{4})', b)
        if m: return [int(m[1], 16)]
        raise ValueError('bad tag ' + t)

def build(name, tl, db):
    src = os.path.join(UNPACKED, name + '.BIZ.lz')
    d = bytearray(open(src, 'rb').read())
    m = charmap(bytes(d), db)
    enc = Encoder(); probs = []; missing = 0
    tops = list(struct.unpack_from('<8I', d, TEXT))
    gs = groups(bytes(d))
    slots = [k for k, t in enumerate(tops) if t]
    # per group: list of (ptr ids, encoded values)
    out_groups = []
    for gi, (to, ptrs) in enumerate(gs):
        segs = []
        starts = sorted(set(ptrs))
        for si, s in enumerate(starts):
            e = starts[si+1] if si + 1 < len(starts) else to
            jp = decode(struct.unpack_from('<%dH' % ((e - s)//2), d, s), m)
            # strip zero padding words that precede the pointer table
            while jp.endswith('{end}{end}'): jp = jp[:-5]
            en = SLOT_EN.get((slots[gi], jp), tl.get(jp))
            ids = [k for k, p in enumerate(ptrs) if p == s]
            if en == '=':
                raw = list(struct.unpack_from('<%dH' % ((e - s)//2), d, s))
                while raw[-2:] == [0, 0]: raw.pop()
                bad = [v for v in raw if FIRST <= v < 0x1000 and v != 0x061e]
                if bad: probs.append(f'{name} g{gi} s{si}: "=" used with kanji glyphs')
                segs.append((ids, raw)); continue
            if en is None:
                missing += 1; en = jp_fallback(jp)
            elif skeleton(en) != skeleton(jp):
                probs.append(f'{name} g{gi} s{si}: tags differ\n  JP {skeleton(jp)}\n  EN {skeleton(en)}')
            jlines = max(p.count('{n}') + 1 for p in jp.split('{p}'))
            en, p = layout(en, max(LINES, jlines))
            probs += [f'{name} g{gi} s{si}: {x}' for x in p]
            for line in re.split(r'\{n\}|\{p\}|\{end\}', en):
                w = cells(line)
                if w > CELLS: probs.append(f'{name} g{gi} s{si}: line {w} cells > {CELLS}: {line}')
            segs.append((ids, enc.encode(en)))
        out_groups.append((len(ptrs), segs))
    if FBASE + (FIRST + len(enc.pairs)) * GB > hook.LF_OFF:
        probs.append(f'{name}: too many tiles ({len(enc.pairs)})')
    # font: keep shared 0..336, append pairs
    nglyph = FIRST + len(enc.pairs)
    font = bytearray(d[:FBASE + FIRST*GB])
    for p in sorted(enc.pairs, key=lambda q: enc.pairs[q]):
        lv = digraph.levels_small(p[3:-1]) if p.startswith('{t:') else digraph.levels(p)
        font += digraph.pack(lv)
    blen = 12 + nglyph * GB
    struct.pack_into('<IHHHH', font, 0x174, blen, 0x200, 0, 5, nglyph * GH)
    out = bytearray(TEXT)
    out[:len(font)] = font
    out[hook.LF_OFF:hook.LF_OFF + len(LETTERS)] = LETTERS
    # text area
    out += bytes(0x20)
    present = [k for k, t in enumerate(tops) if t]
    newtops = [0] * 8
    for slot, (nptr, segs) in zip(present, out_groups):
        table = [0] * nptr
        for ids, vals in segs:
            for k in ids: table[k] = LOAD + len(out)
            out += struct.pack('<%dH' % len(vals), *vals)
        while len(out) % 4: out += b'\0\0'
        newtops[slot] = LOAD + len(out)
        out += struct.pack('<%dI' % nptr, *table)
    struct.pack_into('<8I', out, TEXT, *newtops)
    size = len(out)
    if size > 0x4b000: probs.append(f'{name}: size {size:#x} > 0x4b000')
    out += bytes(0x4b000 - len(out)) if size <= 0x4b000 else b''
    return bytes(out), probs, missing, len(enc.pairs), size

def jp_fallback(jp):
    """Untranslated: keep only the control skeleton plus a marker."""
    return '(untranslated)' + ''.join(t for t in TAG.findall(jp) if t not in ('{n}', '{sp}'))

if __name__ == '__main__':
    tl = load_tl(); db = kanji_db()
    os.makedirs('private/build', exist_ok=True)
    check = '--check' in sys.argv     # report problems only, write nothing
    for name in [a for a in sys.argv[1:] if a != '--check']:
        data, probs, missing, npairs, size = build(name, tl, db)
        for p in probs: print(p)
        if check:
            print(f'{name}: untranslated {missing}, text end {size:#x}, problems {len(probs)}')
            continue
        open(f'private/build/{name}.BIZ.lz', 'wb').write(data)
        c = lz.compress(data)
        open(f'private/build/{name}.BIZ', 'wb').write(c)
        print(f'{name}: {npairs} pairs, text end {size:#x}, untranslated {missing}, '
              f'compressed {len(c)} bytes, problems {len(probs)}')
        if name == 'MBMS13':
            import teamup               # MGFIN Team Up name table
            if os.path.exists(teamup.BUILD): teamup.write(data)
            else: print('MBMS13: run hook.py first, MGFIN Team Up table not written')
