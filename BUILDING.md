# Building the patch

This covers building the patched image from source and editing the
translation. To play, apply the release patch instead (see
[README.md](README.md)).

## Requirements

The build runs on Linux. It needs:

- Python 3 with numpy, scipy and Pillow.
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
