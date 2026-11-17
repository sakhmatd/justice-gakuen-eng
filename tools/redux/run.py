# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Run PCSX-Redux headless with the Lua harness.
Usage: run.py IMAGE.cue STEPSLUA OUTDIR [extra lua file...]
STEPSLUA is Lua source assigning STEPS = {...}.
Each run gets its own copy of the config directory (memory cards, pcsx.json)
in OUTDIR/_cfg, so several runs can go at once. No lock is needed:
patch_bin.py renames a finished image into place, so a running emulator keeps
its open file."""
import os, sys, subprocess, shutil
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REDUX = os.path.join(ROOT, 'private/pcsx-redux/squashfs-root/AppRun')
BIOS = os.path.join(ROOT, 'private/bios/scph5500.bin')
def refresh_filetab(image, outdir):
    """A save state carries the exe file table (sizes) of the build that
    made it. A size that is too large makes the streaming loader decode
    past the end of the file and corrupt it, so after loading a state copy
    the table of IMAGE into RAM at frame 10."""
    lba = next(int(l.split()[0]) for l in open(os.path.join(ROOT, 'private/lba.txt')) if 'SLPS_012.40' in l)
    binf = os.path.splitext(os.path.abspath(image))[0] + '.bin'
    exe = b''
    with open(binf, 'rb') as f:
        for k in range((0x7109c + 842 * 12) // 2048 + 1):
            f.seek((lba + k) * 2352 + 24); exe += f.read(2048)
    tab = os.path.join(os.path.abspath(outdir), '_filetab.bin')
    open(tab, 'wb').write(exe[0x7109c:0x7109c + 842 * 12])
    return (f"\ntable.insert(STEPS, 1, {{10, 'lua', function() local m = PCSX.getMemPtr()\n"
            f"  local tf = io.open('{tab}', 'rb'); local tb = tf:read('*a'); tf:close()\n"
            f"  ffi.copy(m + 0x8089c, tb, #tb) end}})\n")

def run(image, steps, outdir, extra=(), timeout=600):
    os.makedirs(outdir, exist_ok=True)
    pre = os.path.join(os.path.abspath(outdir), '_steps.lua')
    if 'loadSaveState' in steps:
        steps += refresh_filetab(image, outdir)
    open(pre, 'w').write(f'OUTDIR = "{os.path.abspath(outdir)}"\n' + steps)
    cfg = os.path.join(os.path.abspath(outdir), '_cfg')
    shutil.rmtree(cfg, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, 'private/pcsx-redux/cfg'), cfg)
    cmd = [REDUX, '-no-ui', '-portable', cfg, '-safe',
           '-bios', BIOS, '-iso', os.path.abspath(image), '-run', '-noupdate'] + (['-debugger', '-interpreter'] if os.environ.get('REDUX_DEBUG') else []) + [
           '-dofile', pre, '-dofile', os.path.join(ROOT, 'tools/redux/harness.lua')]
    cmd += ['-dofile', os.path.join(ROOT, 'tools/redux/exambot.lua')]
    for e in extra: cmd += ['-dofile', e]
    return subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
def to_png(outdir, name):
    from PIL import Image
    w, h, bpp = map(int, open(os.path.join(outdir, name + '.txt')).read().split())
    raw = open(os.path.join(outdir, name + '.raw'), 'rb').read()
    if bpp == 24 or len(raw) == w*h*3:
        im = Image.frombytes('RGB', (w, h), raw)
    else:
        px = []
        for i in range(0, w*h*2, 2):
            v = raw[i] | raw[i+1] << 8
            px.append(((v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3))
        im = Image.new('RGB', (w, h)); im.putdata(px)
    p = os.path.join(outdir, name + '.png'); im.save(p); return p
if __name__ == '__main__':
    r = run(sys.argv[1], open(sys.argv[2]).read(), sys.argv[3], sys.argv[4:])
    print(r.returncode); print(r.stdout[-3000:]); print(r.stderr[-2000:])
