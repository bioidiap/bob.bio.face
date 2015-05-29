#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""Features for face recognition"""

import bob.ip.base
import numpy

from bob.bio.base.extractor import Extractor

class DCTBlocks (Extractor):

  """Extracts DCT blocks"""
  def __init__(
      self,
      block_size = 12,    # 1 or two parameters for block size
      block_overlap = 11, # 1 or two parameters for block overlap
      number_of_dct_coefficients = 45,
      normalize_blocks = True,
      normalize_dcts = True,
      auto_reduce_coefficients = False
  ):

    # call base class constructor
    Extractor.__init__(
        self,
        block_size = block_size,
        block_overlap = block_overlap,
        number_of_dct_coefficients = number_of_dct_coefficients,
        normalize_blocks = normalize_blocks,
        normalize_dcts = normalize_dcts,
        auto_reduce_coefficients = auto_reduce_coefficients
    )

    # block parameters
    block_size = block_size if isinstance(block_size, (tuple, list)) else (block_size, block_size)
    block_overlap = block_overlap if isinstance(block_overlap, (tuple, list)) else (block_overlap, block_overlap)

    if block_size[0] < block_overlap[0] or block_size[1] < block_overlap[1]:
      raise ValueError("The overlap '%s' is bigger than the block size '%s'. This won't work. Please check your setup!"%(block_overlap, block_size))
    if block_size[0] * block_size[1] <= number_of_dct_coefficients:
      if auto_reduce_coefficients:
        number_of_dct_coefficients = block_size[0] * block_size[1] - 1
      else:
        raise ValueError("You selected more coefficients %d than your blocks have %d. This won't work. Please check your setup!"%(number_of_dct_coefficients, block_size[0] * block_size[1]))

    self.dct_features = bob.ip.base.DCTFeatures(number_of_dct_coefficients, block_size, block_overlap, normalize_blocks, normalize_dcts)

  def __call__(self, image):
    """Computes and returns DCT blocks for the given input image"""
    assert isinstance(image, numpy.ndarray)
    assert image.ndim == 2
    assert image.dtype == numpy.float64

    # Computes DCT features
    return self.dct_features(image)
