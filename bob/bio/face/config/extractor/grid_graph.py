#!/usr/bin/env python

import bob.bio.base
import bob.bio.face
import math

extractor = bob.bio.face.extractor.GridGraph(
    # Gabor parameters
    gabor_sigma = math.sqrt(2.) * math.pi,

    # what kind of information to extract
    normalize_gabor_jets = True,

    # setup of the fixed grid
    node_distance = (8, 8)
)
