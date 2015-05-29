#!/usr/bin/env python

import bob.bio.face
import math

algorithm = bob.bio.face.algorithm.GaborJet(
    # Gabor jet comparison
    gabor_jet_similarity_type = 'PhaseDiffPlusCanberra',
    multiple_feature_scoring = 'max_jet',
    # Gabor wavelet setup
    gabor_sigma = math.sqrt(2.) * math.pi,

)
