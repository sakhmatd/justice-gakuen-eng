#!/bin/sh
# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
# Full rebuild (notes.org order) into $1.bin/.cue (default private/re/test/test).
# Run under: flock private/build/.rebuild.lock sh tools/rebuild_test.sh OUTBASE
set -e
OUT=${1:-private/re/test/test}
cd "$(dirname "$0")/.."
python3 tools/hook.py
python3 tools/build_mbms.py MBMS00 MBMS01 MBMS02 MBMS03 MBMS05 MBMS06 MBMS07 MBMS08 MBMS09 MBMS10 MBMS11 MBMS12 MBMS13 MBMS20 MBMS21 MBMSO04 MBMST04 MBMSP04 MBMSG04
python3 tools/build_menus.py
python3 tools/build_logo.py
python3 tools/build_diarygfx.py
python3 tools/build_entry.py
python3 tools/build_title.py
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
python3 -c "import os,sys; sys.path.insert(0,'tools'); import patch_bin; patch_bin.main(sys.argv[1], {f for f in os.listdir('private/build') if not f.endswith('.lz') and not f.startswith('.')})" "$OUT"
