# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Glyphs 0-336 common to every MBMS font. Tags in {} are non-text symbols."""
S = {}
S[0] = '　'
for i, c in enumerate('1234567890'): S[1+i] = c
for i in range(26): S[11+i] = chr(65+i); S[37+i] = chr(97+i)
for i, c in enumerate('@#%?&♥+*><$!－^'): S[63+i] = c
S[77] = '{77}'; S[78] = '★'; S[79] = '☆'; S[80] = '←'; S[81] = '→'
for i, c in enumerate('。、―=:'): S[82+i] = c
kata = ('アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨ'
        'ラリルレロワヲンヴガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポァィゥェォャュョッ')
for i, c in enumerate(kata): S[87+i] = c
assert 87 + len(kata) == 168
for i, c in enumerate('ー＿，・；‼々-／＼～…‘’“”（）「」『』±×÷♂♀￥○◇※♪α㎜㎝㎞㎏№°″␣'):
    S[168+i] = c
hira = ('あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよ'
        'らりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉっゃゅょ')
for i, c in enumerate(hira): S[209+i] = c
assert 209 + len(hira) == 289, 209 + len(hira)
for i, t in enumerate(['{CIRCLE}', '{SQUARE}', '{CROSS}', '{TRIANGLE}', '{L1}', '{L2}', '{R1}', '{R2}']):
    S[289+i] = t
for i, c in enumerate('山本敏行廣門輝明瀬紀生香川雄一福島淳史染谷勝彦学伊津野英昭舩裕司寺田隆良中江竜也'):
    S[297+i] = c
assert len(S) == 337, len(S)
