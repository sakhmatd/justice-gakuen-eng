# Building the patch

This covers building the patched image from source and editing the
translation. To play, apply the release patch instead (see
[README.md](README.md)).

## Requirements

The build runs on Linux. It needs:

- Python 3 with numpy, scipy, Pillow and OpenCV 4 or later with the
  contrib modules (`cv2.xphoto`, e.g. the pip package
  `opencv-contrib-python`).
- Your dumps of the JP Evolution Disc and of *Rival Schools: United by
  Fate (USA) (Disc 1) (Arcade Disc)* (SLUS-00681). The fight callouts are
  taken from the USA disc.
- The fonts the redraws use, in `/usr/share/fonts/TTF/`: DejaVu Sans
  Bold, DejaVu Sans Condensed Bold, DejaVu Serif, Roboto Condensed Bold,
  Inconsolata ExtraCondensed Black, and the Windows fonts Arial Bold,
  Arial Black, Impact, Times New Roman Italic, Verdana Bold, Tahoma Bold,
  Georgia Bold Italic, Segoe Script Bold, Segoe UI Black Italic, Palatino
  Linotype and Bahnschrift. Other versions of a font can change the
  pixels of the redrawn graphics.

## Text removal (GPU)

The redraws of the event banners, month cards, Festival header and
Festival pictures erase the Japanese text with LaMa (`tools/inpaint.py`).
LaMa needs PyTorch and the weights file `big-lama.pt` (the TorchScript
model from IOPaint, MD5 e3aa4aaa15225a33ec84f9f4bc47e500). The weights
are not in the repository. Put them at `private/models/big-lama.pt`, or
set `LAMA_MODEL` to their path.

- With a CUDA GPU, torch runs LaMa on the GPU.
- Without one, LaMa runs on the CPU. This is slower and gives the same
  kind of result.
- Without torch or the weights, the build falls back to an OpenCV fill
  (method `auto`). It needs no extra files, but it is softer on busy art,
  and those graphics differ from the release.

LaMa output also depends on the GPU, driver and torch version. A build on
another setup can differ from the release image in a few pixels of these
graphics. The v1.1 release was built with torch 2.15 on CUDA and OpenCV
5.1.

## Build

From the repository root:

    python3 tools/setup.py JP.bin USA.bin    # extracts the game data into private/
    sh tools/build_all.sh                    # -> private/out/justice-en.bin/.cue
    python3 tools/make_release.py VERSION    # -> private/dist/justice-en-VERSION.zip

`setup.py` checks both dumps against their known CRC32 values.
Everything it writes is gitignored.

## Translation files

`tl/*.txt` hold the translation as `JP:` / `EN:` pairs keyed by the
exact Japanese text. The first occurrence of a key in any file is used
for every repeat.

- `{p}` and `{end}` must match the Japanese sequence. Other tags may move.
- Text wraps at 20 cells per line and 4 lines per page. Two Latin letters
  share one cell. A colour change starts a new cell.
- Only ASCII and the symbols of the game's font may be used (see
  `tools/shared_chars.py`).

Edit a file, then check it with `python3 tools/build_mbms.py --check
NAME` (for the MBMS files) and rebuild.

`tools/redux/` holds a scripted PCSX-Redux harness that was used for
testing. PCSX-Redux and a BIOS are not included.
