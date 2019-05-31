#!/usr/bin/env python3
# (c) 2019 Yuanzhi Ke
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
1. load the CHISE db pickle dump made in Ke's radicals project
2. load the word vocabulary
3. link the word and the characters in it
4. get the transitive closure of all words, characters and radicals
"""

import pickle
# 1. load the CHISE db
chise_db = pickle.load(open('CHISE_Basic_noShapeDesc_noExpand_noEoc.pkl', 'rb'))
chise_chars, chise_radicals, chise_map = chise_db[0], chise_db[1], chise_db[2]
# clean the chars, radicals list
chise_chars = [x for x in chise_chars if x not in {'<eow>', '<pad>', '<unk>'}]
chise_radicals = [x for x in chise_radicals if x not in {'<eoc>', '<pad>', '<unk>'}]
# 2. load the word vocabulary
# TODO: find a resonable base word vectors and the