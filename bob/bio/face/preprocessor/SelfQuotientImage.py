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
import math
from .Base import Base
from .utils import load_cropper
from bob.bio.base.preprocessor import Preprocessor

class SelfQuotientImage (Base):
  """Crops the face (if desired) and applies self quotient image algorithm [WLW04]_ to photometrically enhance the image.

  **Parameters:**

  face_cropper : str or :py:class:`bob.bio.face.preprocessor.FaceCrop` or :py:class:`bob.bio.face.preprocessor.FaceDetect` or ``None``
    The face image cropper that should be applied to the image.
    If ``None`` is selected, no face cropping is performed.
    Otherwise, the face cropper might be specified as a registered resource, a configuration file, or an instance of a preprocessor.

    .. note:: The given class needs to contain a ``crop_face`` method.

  sigma : float
    Please refer to the [WLW04]_ original paper (see :py:class:`bob.ip.base.SelfQuotientImage` documentation).

  kwargs
    Remaining keyword parameters passed to the :py:class:`Base` constructor, such as ``color_channel`` or ``dtype``.
  """

  def __init__(
      self,
      face_cropper,
      sigma = math.sqrt(2.),
      **kwargs
  ):

    Base.__init__(self, **kwargs)

    # call base class constructor with its set of parameters
    Preprocessor.__init__(
        self,
        face_cropper = face_cropper,
        sigma = sigma
    )

    self.cropper = load_cropper(face_cropper)

    size = max(1, int(3. * sigma))
    self.self_quotient = bob.ip.base.SelfQuotientImage(size_min = size, sigma = sigma)


  def __call__(self, image, annotations = None):
    """__call__(image, annotations = None) -> face

    Aligns the given image according to the given annotations.

    First, the desired color channel is extracted from the given image.
    Afterward, the face is eventually cropped using the ``face_cropper`` specified in the constructor.
    Then, the image is photometrically enhanced using the self quotient image algorithm [WLW04]_.
    Finally, the resulting face is converted to the desired data type.

    **Parameters:**

    image : 2D or 3D :py:class:`numpy.ndarray`
      The face image to be processed.

    annotations : dict or ``None``
      The annotations that fit to the given image.
      Might be ``None``, when the ``face_cropper`` is ``None`` or of type :py:class:`FaceDetect`.

    **Returns:**

    face : 2D :py:class:`numpy.ndarray`
      The cropped and photometrically enhanced face.
    """
    image = self.color_channel(image)
    if self.cropper is not None:
      image = self.cropper.crop_face(image, annotations)
    image = self.self_quotient(image)
    return self.data_type(image)
