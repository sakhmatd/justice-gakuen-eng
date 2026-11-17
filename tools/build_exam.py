# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Rebuild exam text images (TESTMJxx.TIZ) and SIKEN.BIN layout tables.

Reads tl/EXAM.txt. Writes private/build/TESTMJxx.TIZ (+ .lz) and
private/build/SIKEN.BIN. Every message line and command gets its own run of
32x16 cells; scripts are rewritten to match.
"""
import sys, os, re, struct
sys.path.insert(0, os.path.dirname(__file__))
from exam import load, msg_script, cmd_structs, B
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import lz

FONT = '/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf'
MAXW = 14 * 32                       # widest bubble line, px
MAXLINES = 5
TITLE_W = 0x8009d7d8                 # HUD title widths, u8 x 30
MSG_AREA = (0x8009c040, 0x8009c1f8)  # message scripts, then the pointer table
BG = 14                              # set per image by use_palette()
RAMP = list(range(1, 14))            # dark -> light text indices
TFILL, TLINE = 0, 1                  # title fill / outline

def parse(path='tl/EXAM.txt'):
    lessons, cmds, cur = {}, {}, None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line or line.startswith('#'): continue
        m = re.fullmatch(r'\[(\d+|cmds)\]', line)
        if m: cur = m[1]; continue
        if cur == 'cmds':
            k, v = line.split(':', 1); cmds[int(k)] = v.strip(); continue
        k, v = line.split(':', 1)
        lessons.setdefault(int(cur) - 1, {})[k.strip()] = v.strip()
    return lessons, cmds

def read_tim(L):
    d = open(f'private/unpacked/TESTMJ{L+1:02d}.TIZ.lz', 'rb').read()
    bl = struct.unpack_from('<I', d, 8)[0]; p = 8 + bl
    blen, x, y, w, h = struct.unpack_from('<IHHHH', d, p)
    raw = np.frombuffer(d[p+12:p+blen], np.uint8)
    px = np.empty(raw.size * 2, np.uint8); px[0::2] = raw & 15; px[1::2] = raw >> 4
    return d[:p+12], px.reshape(h, w*4)

def pack_tim(head, px):
    f = px.ravel()
    return head + bytes((f[0::2] | (f[1::2] << 4)).astype(np.uint8))

_fonts = {}
def font(size):
    if size not in _fonts: _fonts[size] = ImageFont.truetype(FONT, size)
    return _fonts[size]

def lum(v): return (v & 31) + (v >> 5 & 31) + (v >> 10 & 31)

def use_palette(head, orig, start):
    """Derive background, text ramp and title colours from the original."""
    global BG, RAMP, TFILL, TLINE
    import collections
    clut = struct.unpack_from('<16H', head, 20)
    body = orig[16:, :]
    cnt = collections.Counter(body.ravel().tolist())
    BG = cnt.most_common(1)[0][0]
    def grey(v):
        r, g, b = v & 31, v >> 5 & 31, v >> 10 & 31
        return max(r, g, b) - min(r, g, b) <= 2
    used = [i for i, n in cnt.items() if i != BG and n > 8 and grey(clut[i])]
    RAMP = sorted(used, key=lambda i: lum(clut[i]))
    tcnt = collections.Counter(orig[0:16, 0:min(start, 8)*32].ravel().tolist())
    tused = [i for i, n in tcnt.items() if i != BG and n > 8]
    TFILL = max(tused, key=lambda i: lum(clut[i])); TLINE = min(tused, key=lambda i: lum(clut[i]))

def alpha_to_idx(a):
    r = np.array(RAMP, np.uint8)
    k = np.clip(np.rint((1 - a) * (len(r) - 1)), 0, len(r) - 1).astype(int)
    return np.where(a > 0.12, r[k], BG).astype(np.uint8)

def text_width(s, size=13):
    return font(size).getlength(s)

def render_line(text, orig, size=13):
    """Render one line (with {c:N} icon tokens) to a 16-row index strip."""
    parts = re.split(r'(\{c:\d+\})', text)
    w = 0; segs = []
    for p in parts:
        if not p: continue
        m = re.fullmatch(r'\{c:(\d+)\}', p)
        if m:
            n = int(m[1]); cell = orig[(n//8)*16:(n//8)*16+16, (n%8)*32:(n%8)*32+32]
            cols = np.nonzero((cell != BG).any(axis=0))[0]
            cell = cell[:, :cols.max()+2] if len(cols) else cell[:, :8]
            segs.append(('img', cell)); w += cell.shape[1] + 2
        else:
            tw = int(np.ceil(font(size).getlength(p))) + 1
            segs.append(('txt', p, tw)); w += tw
    out = np.full((16, max(1, w)), BG, np.uint8); x = 0
    for s in segs:
        if s[0] == 'img':
            c = s[1]; out[:, x:x+c.shape[1]] = c; x += c.shape[1] + 2
        else:
            im = Image.new('L', (s[2], 16)); ImageDraw.Draw(im).text((0, 0), s[1], font=font(size), fill=255)
            out[:, x:x+s[2]] = alpha_to_idx(np.asarray(im, np.float32) / 255); x += s[2]
    return out

TITLE_SIZE, TITLE_BASE = 13, 11       # HUD title: caps on rows 1-10 like the digit

def title_alpha(text, size, ss=4):
    """Alpha (15, w) of the title, baseline at row TITLE_BASE, cropped to ink."""
    f = font(size * ss)
    W = (int(f.getlength(text)) // ss + 8) * ss
    im = Image.new('L', (W, 15 * ss))
    ImageDraw.Draw(im).text((4 * ss, TITLE_BASE * ss), text, font=f, fill=255, anchor='ls')
    a = np.asarray(im.resize((W // ss, 15), Image.BOX), np.float32) / 255
    cols = np.nonzero((a > 0.05).any(0))[0]
    return a[:, cols.min():cols.max() + 1]

def render_title(text, width):
    """HUD title strip (16 rows): white fill, 1 px dark outline. The HUD
    draws 15 rows (SIKEN 0x800a2a24), so row 15 stays empty. The outline
    is a plain dilation; the old np.roll version wrapped clipped
    descenders into row 0 as dark bars."""
    size = TITLE_SIZE
    while True:
        a = title_alpha(text, size)
        if a.shape[1] + 2 <= width or size <= 8: break
        size -= 1
    a = np.pad(a, ((0, 1), (1, 1)))
    fill = a > 0.45
    dil = ndimage.binary_dilation(fill, np.ones((3, 3), bool))
    assert not fill[0].any() and not dil[15].any(), text
    return np.where(fill, TFILL, np.where(dil, TLINE, BG)).astype(np.uint8)

def wrap(msg):
    lines = []
    for para in msg.split('|'):
        cur = ''
        for word in para.split(' '):
            cand = (cur + ' ' + word) if cur else word
            vis = re.sub(r'\{c:\d+\}', 'MMM', cand)
            if cur and text_width(vis) > MAXW - 4: lines.append(cur); cur = word
            else: cur = cand
        lines.append(cur)
    return lines

class Cells:
    def __init__(self, px, start):
        self.px = px; self.next = start
    def put(self, strip):
        n = (strip.shape[1] + 31) // 32
        first = self.next
        if first + n > 128: raise ValueError('out of cells')
        pad = np.full((16, n*32), BG, np.uint8); pad[:, :strip.shape[1]] = strip
        for k in range(n):
            c = first + k
            self.px[(c//8)*16:(c//8)*16+16, (c%8)*32:(c%8)*32+32] = pad[:, k*32:k*32+32]
        self.next += n
        return first, n

def build():
    lessons, cmdtext = parse()
    c = bytearray(load())
    probs = []
    scripts = []
    for L in range(30):
        tr = lessons[L]
        head, orig = read_tim(L)
        _, olines, _ = msg_script(bytes(c), L)
        start = min(l[0][0] for l in olines if l)
        use_palette(head, orig, start)
        px = np.full_like(orig, BG)
        px[0:16, 0:start*32 if start <= 8 else 256] = BG
        # title: in its original cells (row 0; the HUD draws only row 0).
        # The HUD width table (SIKEN 0x17d8, one byte per lesson) gets the
        # strip width so no other cell shows next to it.
        t = render_title(tr['title'], min(start, 8) * 32)
        px[0:16, 0:t.shape[1]] = t
        c[TITLE_W - B + L] = t.shape[1]
        cells = Cells(px, start)
        lines = wrap(tr['msg'])
        if len(lines) > MAXLINES: probs.append(f'L{L+1}: {len(lines)} lines')
        script = bytearray()
        for i, ln in enumerate(lines):
            strip = render_line(ln, orig)
            if strip.shape[1] > MAXW: probs.append(f'L{L+1}: line too wide: {ln}')
            first, n = cells.put(strip)
            if i: script.append(0xfe)
            script += bytes([first, n])
        script.append(0xff)
        scripts.append(script)
        sp, order, cmds = cmd_structs(bytes(c), L)
        addrs = sorted(q for q, _, _ in cmds)
        for q, param, cl in cmds:
            if not any(cl): continue
            txt = cmdtext.get(param, '?')
            parts = []
            for i, ln in enumerate(txt.split('|')):
                first, n = cells.put(render_line(ln, orig))
                parts.append((first, n))
            body = bytearray([param, len(parts)])
            for i, (f, n) in enumerate(parts):
                if i: body.append(0xff)
                body += bytes([f, n])
            nxt = [a for a in addrs if a > q]
            limit = (nxt[0] if nxt else sp) - q
            if len(body) > limit:
                probs.append(f'L{L+1}: cmd p{param} needs {len(body)} > {limit} bytes')
                continue
            c[q - B:q - B + len(body)] = body
        tim = pack_tim(head, px)
        open(f'private/build/TESTMJ{L+1:02d}.TIZ.lz', 'wb').write(tim)
        open(f'private/build/TESTMJ{L+1:02d}.TIZ', 'wb').write(lz.compress(tim))
        Image.fromarray(np.where(px == BG, 40, 255 - px * 18).astype(np.uint8)).save(f'private/testmj/new{L+1:02d}.png')
    # repack message scripts
    lo, hi = MSG_AREA
    blob = bytearray(); ptrs = []
    for s in scripts:
        ptrs.append(lo + len(blob)); blob += s
    if len(blob) > hi - lo: probs.append(f'message scripts {len(blob)} > {hi - lo} bytes')
    else:
        c[lo - B:lo - B + len(blob)] = blob
        c[lo - B + len(blob):hi - B] = bytes(hi - lo - len(blob))
        for L, p in enumerate(ptrs): struct.pack_into('<I', c, 0x8009c1f8 - B + 4*L, p)
    open('private/build/SIKEN.BIN', 'wb').write(c)
    for p in probs: print(p)
    print(f'exam: 30 images, scripts {len(blob)}/{hi - lo} bytes, problems {len(probs)}')

if __name__ == '__main__':
    build()
