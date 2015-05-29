#!/usr/bin/env python

import bob.bio.face

extractor = bob.bio.face.extractor.DCTBlocks(
    block_size = 12,
    block_overlap = 11,
    number_of_dct_coefficients = 45
)
