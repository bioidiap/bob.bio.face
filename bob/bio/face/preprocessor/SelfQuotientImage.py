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
  """Crops the face according to the eye positions (if given), computes the self quotient image."""

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
    self.sqi = bob.ip.base.SelfQuotientImage(size_min = size, sigma = sigma)


  def self_quotient(self, image):
    return self.sqi(image)


  def __call__(self, image, annotations = None):
    """Crops the face using the specified face cropper and performs Self-Quotient Image preprocessing."""
    image = self.color_channel(image)
    if self.cropper is not None:
      image = self.cropper.crop_face(image, annotations)
    image = self.self_quotient(image)
    return self.data_type(image)
