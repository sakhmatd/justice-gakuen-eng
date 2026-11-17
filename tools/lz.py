# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Word-based LZ used by the game (exe 0x80015dbc).

Stream of u16 LE. Flag word, 16 items MSB-first. Bit 0: copy one literal
word. Bit 1: ref word v; len = v >> 11, dist = v & 0x7ff (halfwords); if
len == 0 the next word is len. dist == 0 and len == 0 ends the stream.
"""
import struct, sys

def decompress(src, pos=0):
    out = bytearray()
    n = len(src)
    while pos + 1 < n:
        flags = src[pos] | src[pos+1] << 8; pos += 2
        for _ in range(16):
            if not flags & 0x8000:
                out += src[pos:pos+2]; pos += 2
            else:
                v = src[pos] | src[pos+1] << 8; pos += 2
                ln, dist = v >> 11, v & 0x7ff
                if ln == 0:
                    ln = src[pos] | src[pos+1] << 8; pos += 2
                if dist == 0 and ln == 0:
                    return bytes(out), pos
                p = len(out) - dist * 2
                for _ in range(ln):
                    out += out[p:p+2]; p += 2
            flags = (flags << 1) & 0xffff
    return bytes(out), pos

if __name__ == '__main__':
    d = open(sys.argv[1], 'rb').read()
    out, end = decompress(d)
    print(f'in {len(d)} consumed {end} out {len(out)}', file=sys.stderr)
    open(sys.argv[2], 'wb').write(out)

def compress(data, chain=64, burst=0xe00):
    """Greedy LZ matching the game's format. data length must be even.

    The game decodes while the CD streams into an 8-sector ring, inside the
    read callback. Output bursts that are too large per input sector let the
    ring overrun (original files stay below ~0x1840 bytes per sector). So no
    0x800-byte input sector may produce more than `burst` output bytes,
    except in the trailing all-zero tail, which decodes after the last read.
    """
    import array
    assert len(data) % 2 == 0
    w = array.array('H', data)
    n = len(w)
    tail = n
    while tail > 0 and w[tail-1] == 0: tail -= 1
    head = {}; prev = [-1] * n
    def insert(i):
        if i + 1 < n:
            k = w[i] | w[i+1] << 16
            prev[i] = head.get(k, -1); head[k] = i
    items = []                       # (is_ref, words)
    cbytes = 0                       # compressed size so far
    per = {}                         # input sector -> output bytes
    def add(r, ws, outb):
        nonlocal cbytes
        if len(items) % 16 == 0: cbytes += 2
        sec = cbytes // 0x800
        per[sec] = per.get(sec, 0) + outb
        cbytes += 2 * len(ws)
        items.append((r, ws))
    def room():
        return burst - per.get((cbytes + (2 if len(items) % 16 == 0 else 0)) // 0x800, 0)
    i = 0
    while i < n:
        best_len, best_d = 0, 0
        cap = 0xffff if i >= tail else min(0xffff, room() // 2, max(2, tail - i))
        if i + 1 < n and cap >= 2:
            j = head.get(w[i] | w[i+1] << 16, -1); c = 0
            while j >= 0 and i - j <= 0x7ff and c < chain:
                L = 0
                while i + L < n and w[j+L] == w[i+L] and L < cap: L += 1
                if L > best_len: best_len, best_d = L, i - j
                if best_len >= cap: break
                j = prev[j]; c += 1
        if best_len >= 2:
            ws = [(best_len << 11) | best_d] if best_len <= 31 else [best_d, best_len]
            add(1, ws, best_len * 2)
            for k in range(best_len): insert(i + k)
            i += best_len
        else:
            add(0, [w[i]], 2); insert(i); i += 1
    items.append((1, [0, 0]))
    out = array.array('H')
    for s0 in range(0, len(items), 16):
        grp = items[s0:s0+16]; f = 0
        for k, (r, _) in enumerate(grp):
            if r: f |= 0x8000 >> k
        out.append(f)
        for _, ws in grp: out.extend(ws)
    return out.tobytes()
