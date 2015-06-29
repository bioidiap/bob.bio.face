#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""Features for face recognition"""

import bob.ip.base
import numpy

from bob.bio.base.extractor import Extractor

class DCTBlocks (Extractor):

  """Extracts *Discrete Cosine Transform* (DCT) features from (overlapping) image blocks.
  These features are based on the :py:class:`bob.ip.base.DCTFeatures` class.
  The default parametrization is the one that performed best on the BANCA database in [WMM+11]_.

  Usually, these features are used in combination with the algorithms defined in :ref:`bob.bio.gmm <bob.bio.gmm>`.
  However, you can try to use them with other algorithms.

  **Parameters:**

  block_size : int or (int, int)
    The size of the blocks that will be extracted.
    This parameter might be either a single integral value, or a pair ``(block_height, block_width)`` of integral values.

  block_overlap : int or (int, int)
    The overlap of the blocks in vertical and horizontal direction.
    This parameter might be either a single integral value, or a pair ``(block_overlap_y, block_overlap_x)`` of integral values.
    It needs to be smaller than the ``block_size``.

  number_of_dct_coefficients : int
    The number of DCT coefficients to use.
    The actual number will be one less since the first DCT coefficient (which should be 0, if normalization is used) will be removed.

  normalize_blocks : bool
    Normalize the values of the blocks to zero mean and unit standard deviation before extracting DCT coefficients.

  normalize_dcts : bool
    Normalize the values of the DCT components to zero mean and unit standard deviation. Default is ``True``.
  """
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
    """__call__(image) -> feature

    Computes and returns DCT blocks for the given input image.

    **Parameters:**

    image : 2D :py:class:`numpy.ndarray` (floats)
      The image to extract the features from.

    **Returns:**

    feature : 2D :py:class:`numpy.ndarray` (floats)
      The extracted DCT features for all blocks inside the image.
      The first index is the block index, while the second index is the DCT coefficient.
    """
    assert isinstance(image, numpy.ndarray)
    assert image.ndim == 2
    assert image.dtype == numpy.float64

    # Computes DCT features
    return self.dct_features(image)

  # re-define the train function to get it non-documented
  def train(*args,**kwargs) : raise NotImplementedError("This function is not implemented and should not be called.")
  def load(*args,**kwargs) : pass
