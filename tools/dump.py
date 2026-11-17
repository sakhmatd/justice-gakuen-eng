# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Dump MBMS text to private/script/<NAME>.txt.

Each group's string block is split at every pointer target into segments.
Header line: '## <file> g<group> s<seg> @<offset> ids=<pointer indices>'.
Tags: {n} newline 0x3000, {p} page end 0x3001, {sp} 0x061e, {face:x} 0x3002,
{col:x} 0x3003 (x=- for 255), {v0}..{v9} or {vN:arg} 0x200N, {end} 0x0000,
{g:N} unknown glyph, {x:HHHH} other.
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
from mbms import *
from shared_chars import S

ARG = {0x2003, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x3002, 0x3003}

def kanji_db():
    db = {}
    for l in open('data/kanji.tsv'):
        g, r, o = l.rstrip('\n').split('\t'); db[g] = r
    return db

def charmap(d, db):
    m = dict(S)
    for i in range(337, font_count(d)):
        g = glyph(d, i)
        if any(g): m[i] = db.get(gid(g), '{g:%d}' % i)
    return m

def decode(vals, m):
    out = []; i = 0
    while i < len(vals):
        v = vals[i]; i += 1
        if v == 0: out.append('{end}')
        elif v == 0x3000: out.append('{n}')
        elif v == 0x3001: out.append('{p}')
        elif v == 0x061e: out.append('{sp}')
        elif v in ARG:
            a = vals[i] if i < len(vals) else 0; i += 1
            if v == 0x3002: out.append('{face:%d}' % a)
            elif v == 0x3003: out.append('{col:%s}' % ('-' if a == 255 else a))
            else: out.append('{v%d:%d}' % (v - 0x2000, a))
        elif 0x2000 <= v <= 0x2009: out.append('{v%d}' % (v - 0x2000))
        elif v in m: out.append(m[v])
        else: out.append('{x:%04x}' % v)
    return ''.join(out)

def segments(d):
    """Yield (group, seg, start, end, ids) per group in file order."""
    gs = groups(d)
    for gi, (to, ptrs) in enumerate(gs):
        starts = sorted(set(ptrs))
        for si, s in enumerate(starts):
            e = starts[si+1] if si + 1 < len(starts) else to
            ids = [k for k, p in enumerate(ptrs) if p == s]
            yield gi, si, s, e, ids

def dump(fn, db, out):
    d = open(fn, 'rb').read(); m = charmap(d, db); name = os.path.basename(fn).split('.')[0]
    with open(out, 'w') as o:
        for gi, si, s, e, ids in segments(d):
            vals = struct.unpack_from('<%dH' % ((e - s)//2), d, s)
            # trailing zero padding before the pointer table
            o.write(f'## {name} g{gi} s{si:04d} @{s:x} ids={",".join(map(str, ids))}\n')
            o.write(decode(vals, m) + '\n')

if __name__ == '__main__':
    db = kanji_db(); os.makedirs('private/script', exist_ok=True)
    for fn in files():
        name = os.path.basename(fn).split('.')[0]
        dump(fn, db, f'private/script/{name}.txt')
