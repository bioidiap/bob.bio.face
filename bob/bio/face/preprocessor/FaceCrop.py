#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu May 24 10:41:42 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import bob.ip.base
import numpy

from .Base import Base
from bob.bio.base.preprocessor import Preprocessor

class FaceCrop (Base):
  """Crops the face according to the given annotations"""

  def __init__(
      self,
      cropped_image_size,        # resolution of the cropped image, in order (HEIGHT,WIDTH); if not given, no face cropping will be performed
      cropped_positions,         # dictionary of the cropped positions, usually: {'reye':(RIGHT_EYE_Y, RIGHT_EYE_X) , 'leye':(LEFT_EYE_Y, LEFT_EYE_X)}
      fixed_positions = None,    # dictionary of FIXED positions in the original image; if specified, annotations from the database will be ignored
      mask_sigma = None,         # The sigma for random values areas outside image
      mask_neighbors = 5,        # The number of neighbors to consider while extrapolating
      mask_seed = None,          # The seed for generating random values during extrapolation
      **kwargs                   # parameters to be written in the __str__ method
  ):
    """Parameters of the constructor of this preprocessor:

    cropped_image_size : (int, int)
      The size of the resulting cropped images.

    cropped_positions : dict
      The coordinates in the cropped image, where the annotated points should be put to.
      This parameter is a dictionary with usually two elements, e.g., ``{'reye':(RIGHT_EYE_Y, RIGHT_EYE_X) , 'leye':(LEFT_EYE_Y, LEFT_EYE_X)}``.
      However, also other parameters, such as ``{'topleft' : ..., 'bottomright' : ...}`` are supported, as long as the ``annotations`` in the :py:meth:`__call__` function are present.

    fixed_positions : dict or None
      If specified, ignore the annotations from the database and use these fixed positions throughout.

    mask_sigma : float or None
      Fill the area outside of image boundaries with random pixels from the border, by adding noise to the pixel values.
      To disable extrapolation, set this value to None.
      To disable adding random noise, set it to a negative value or 0.

    mask_neighbors : int
      The number of neighbors used during mask extrapolation.
      See :py:func:`bob.ip.base.extrapolate_mask` for details.

    mask_seed : int or None
      The random seed to apply for mask extrapolation.

      .. warning::
         When run in parallel, the same random seed will be applied to all parallel processes.
         Hence, results of parallel execution will differ from the results in serial execution.

    kwargs
      Remaining keyword parameters passed to the :py:class:`Base` constructor, such as ``color_channel`` or ``dtype``.
    """

    Base.__init__(self, **kwargs)

    # call base class constructor
    Preprocessor.__init__(
        self,
        cropped_image_size = cropped_image_size,
        cropped_positions = cropped_positions,
        fixed_positions = fixed_positions,
        mask_sigma = mask_sigma,
        mask_neighbors = mask_neighbors,
        mask_seed = mask_seed
    )

    # check parameters
    assert len(cropped_positions) == 2
    if fixed_positions:
      assert len(fixed_positions) == 2

    # copy parameters
    self.cropped_image_size = cropped_image_size
    self.cropped_positions = cropped_positions
    self.cropped_keys = sorted(cropped_positions.keys())
    self.fixed_positions = fixed_positions
    self.mask_sigma = mask_sigma
    self.mask_neighbors = mask_neighbors
    self.mask_rng = bob.core.random.mt19937(mask_seed) if mask_seed is not None else bob.core.random.mt19937()

    # create objects required for face cropping
    self.cropper = bob.ip.base.FaceEyesNorm(crop_size=cropped_image_size, right_eye=cropped_positions[self.cropped_keys[0]], left_eye=cropped_positions[self.cropped_keys[1]])
    self.cropped_mask = numpy.ndarray(cropped_image_size, numpy.bool)


  def crop_face(self, image, annotations = None):
    """Executes the face cropping on the given image and returns the cropped version of it"""
    if self.fixed_positions is not None:
      annotations = self.fixed_positions
    if annotations is None:
      raise ValueError("Cannot perform image cropping since annotations are not given, and no fixed annotations are specified.")

    assert isinstance(annotations, dict)
    if not all(k in annotations for k in self.cropped_keys):
      raise ValueError("At least one of the expected annotations '%s' are not given in '%s'." % (self.cropped_keys, annotations.keys()))

    # create output
    mask = numpy.ones(image.shape, dtype=numpy.bool)
    cropped_image = numpy.zeros(self.cropped_image_size)
    self.cropped_mask[:] = False

    # perform the cropping
    self.cropper(
        image,  # input image
        mask,   # full input mask
        cropped_image, # cropped image
        self.cropped_mask,  # cropped mask
        right_eye = annotations[self.cropped_keys[0]], # position of first annotation, usually right eye
        left_eye = annotations[self.cropped_keys[1]]  # position of second annotation, usually left eye
    )

    if self.mask_sigma is not None:
      # extrapolate the mask so that pixels outside of the image original image region are filled with border pixels
      bob.ip.base.extrapolate_mask(self.cropped_mask, cropped_image, self.mask_sigma, self.mask_neighbors, self.mask_rng)

    return cropped_image


  def __call__(self, image, annotations = None):
    """Aligns the given image according to the given annotations."""
    # convert to the desired color channel
    image = self.color_channel(image)
    # crop face
    image = self.crop_face(image, annotations)
    # convert data type
    return self.data_type(image)
