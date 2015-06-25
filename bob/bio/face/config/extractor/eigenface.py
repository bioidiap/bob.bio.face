#!/usr/bin/env python

import bob.bio.face

# compute eigenfaces using the training database
extractor = bob.bio.face.extractor.Eigenface(
    subspace_dimension = .95
)
