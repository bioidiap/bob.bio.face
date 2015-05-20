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
from .utils import load_cropper
from bob.bio.base.preprocessor import Preprocessor

class TanTriggs (Base):
  """Crops the face (if desired) and applies Tan&Triggs algorithm"""

  def __init__(
      self,
      face_cropper,
      gamma = 0.2,
      sigma0 = 1,
      sigma1 = 2,
      size = 5,
      threshold = 10.,
      alpha = 0.1,
      **kwargs
  ):

    """Parameters of the constructor of this preprocessor:

    cropper : str or `bob.bio.face.preprocessor.FaceCrop` or `bob.bio.face.preprocessor.FaceDetect`
      The face image cropper that should be applied to the image.
      It might be specified as a registered resource, a configuration file, or an instance of a preprocessor.

      .. note:: The given class needs to contain a ``crop_face`` method.

    gamma, sigma0, sigma1, size, threshold, alpha
      Please refer to the [TT10]_ original paper (see :py:func:`bob.ip.base.TanTriggs` documentation).

    kwargs
      Remaining keyword parameters passed to the :py:class:`Base` constructor, such as ``color_channel`` or ``dtype``.
    """

    Base.__init__(self, **kwargs)

    # call base class constructor with its set of parameters
    Preprocessor.__init__(
        self,
        face_cropper = face_cropper,
        gamma = gamma,
        sigma0 = sigma0,
        sigma1 = sigma1,
        size = size,
        threshold = threshold,
        alpha = alpha
    )

    self.cropper = load_cropper(face_cropper)
    self.tan_triggs = bob.ip.base.TanTriggs(gamma, sigma0, sigma1, size, threshold, alpha)


  def __call__(self, image, annotations = None):
    """Crops the face using the specified face cropper and performs Tan&Triggs preprocessing."""
    image = self.color_channel(image)
    if self.cropper is not None:
      image = self.cropper.crop_face(image, annotations)
    image = self.tan_triggs(image)
    return self.data_type(image)
