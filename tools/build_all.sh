#!/bin/sh
# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
# Full rebuild: private/build, then the patched image OUTBASE.bin/.cue
# (default private/out/justice-en). Run from the project root after
# tools/setup.py. Two rebuilds must not run at once:
#   flock private/build/.rebuild.lock sh tools/build_all.sh [OUTBASE]
set -e
cd "$(dirname "$0")/.."
mkdir -p private/build private/out private/testmj
python3 tools/hook.py
python3 tools/build_mbms.py MBMS00 MBMS01 MBMS02 MBMS03 MBMS05 MBMS06 MBMS07 MBMS08 MBMS09 MBMS10 MBMS11 MBMS12 MBMS13 MBMS20 MBMS21 MBMSO04 MBMST04 MBMSP04 MBMSG04
python3 tools/build_menus.py
python3 tools/build_logo.py
python3 tools/build_diarygfx.py
python3 tools/build_entry.py
python3 tools/build_title.py
python3 tools/build_titlescreen.py
python3 tools/build_sysmsg.py
python3 tools/build_banners.py
python3 tools/build_exam.py
python3 tools/build_examgfx.py
python3 tools/build_logotile.py
python3 tools/nick.py
python3 tools/build_aload.py
python3 tools/build_fest.py
python3 tools/build_festpics.py
python3 tools/build_staff.py
python3 tools/build_fight.py
python3 tools/build_pause.py
python3 tools/build_editcm.py
python3 tools/build_kanri.py
python3 tools/build_kouka.py
python3 -c "import sys; sys.path.insert(0, 'tools'); import patch_bin; patch_bin.main(sys.argv[1] if len(sys.argv) > 1 else None)" "$@"
