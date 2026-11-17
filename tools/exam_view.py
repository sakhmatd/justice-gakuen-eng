# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Strips of each lesson's message lines and commands, for transcription."""
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))
from exam import *
from PIL import Image, ImageDraw
def cells_img(L, pairs):
    im = Image.open(f'private/testmj/TESTMJ{L+1:02d}.png').convert('RGB')
    out = Image.new('RGB', (32*sum(n for _, n in pairs), 16))
    x = 0
    for cell, n in pairs:
        for k in range(n):
            cc = cell + k
            out.paste(im.crop(((cc%8)*32, (cc//8)*16, (cc%8)*32+32, (cc//8)*16+16)), (x, 0)); x += 32
    return out
c = load()
lo, hi = int(sys.argv[1]), int(sys.argv[2])
rows = []
for L in range(lo-1, hi):
    _, lines, _ = msg_script(c, L)
    rows.append((f'{L+1:02d} title', Image.open(f'private/testmj/TESTMJ{L+1:02d}.png').convert('RGB').crop((0, 0, 32*min(l[0][0] for l in lines if l), 16))))
    for i, ln in enumerate(lines): rows.append((f'{L+1:02d} m{i}', cells_img(L, ln)))
    _, _, cmds = cmd_structs(c, L)
    for i, (q, p, cl) in enumerate(cmds):
        for j, l in enumerate(cl):
            if not l: continue
            rows.append((f'{L+1:02d} c{i} p{p}' + (f' L{j}' if len(cl) > 1 else ''), cells_img(L, l)))
S = 2
W = 90 + max(r[1].width for r in rows) * S
out = Image.new('RGB', (W, len(rows) * (16*S + 4)), (30, 30, 30)); d = ImageDraw.Draw(out)
for i, (lab, im) in enumerate(rows):
    y = i * (16*S + 4); d.text((2, y + 8), lab, fill=(255, 255, 0))
    out.paste(im.resize((im.width*S, 16*S), Image.NEAREST), (90, y))
out.save(sys.argv[3])
