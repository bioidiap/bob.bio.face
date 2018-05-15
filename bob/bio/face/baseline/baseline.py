#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
Defining some face recognition baselines
"""

from bob.bio.base.baseline import Baseline

eigenface = Baseline(name="eigenface",
                     preprocessors={'default': 'face-crop-eyes', 'atnt': 'base'},
                     extractor='linearize',
                     algorithm='pca')

lda = Baseline(name="lda",
               preprocessors={'default': 'face-crop-eyes', 'atnt': 'base'},
               extractor='eigenface',
               algorithm='lda')

plda = Baseline(name="plda",
                preprocessors={'default': 'face-crop-eyes', 'atnt': 'base'},
                extractor='linearize',
                algorithm='pca+plda')


gabor_graph = Baseline(name="gabor_graph",
                       preprocessors={'default': 'inorm-lbp-crop', 'atnt': 'inorm-lbp'},
                       extractor='grid-graph',
                       algorithm='gabor-jet')

lgbphs = Baseline(name="lgbphs",
                  preprocessors={'default': 'tan-triggs-crop', 'atnt': 'tan-triggs'},
                  extractor='lgbphs',
                  algorithm='histogram')

gmm = Baseline(name="gmm",
               preprocessors={'default': 'tan-triggs-crop', 'atnt': 'tan-triggs'},
               extractor='dct-blocks',
               algorithm='gmm')

isv = Baseline(name="isv",
               preprocessors={'default': 'tan-triggs-crop', 'atnt': 'tan-triggs'},
               extractor='dct-blocks',
               algorithm='isv')

ivector = Baseline(name="gmm",
                   preprocessors={'default': 'tan-triggs-crop', 'atnt': 'tan-triggs'},
                   extractor='dct-blocks',
                   algorithm='ivector-cosine')

bic = Baseline(name="bic",
               preprocessors={'default': 'face-crop-eyes', 'atnt': 'base'},
               extractor='grid-graph',
               algorithm='bic-jets')
