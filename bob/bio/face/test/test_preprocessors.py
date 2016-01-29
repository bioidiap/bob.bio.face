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


import unittest
import os
import numpy

from nose.plugins.skip import SkipTest

import pkg_resources

regenerate_refs = False

import bob.bio.base
import bob.bio.face
import bob.db.verification.utils


def _compare(data, reference, write_function = bob.bio.base.save, read_function = bob.bio.base.load, atol = 1e-5, rtol = 1e-8):
  # write reference?
  if regenerate_refs:
    write_function(data, reference)

  # compare reference
  reference = read_function(reference)
  assert numpy.allclose(data, reference, atol=atol, rtol=rtol)
  return reference


def _image():
  return bob.io.base.load(pkg_resources.resource_filename('bob.bio.face.test', 'data/testimage.jpg'))

def _annotation():
  return bob.db.verification.utils.read_annotation_file(pkg_resources.resource_filename('bob.bio.face.test', 'data/testimage.pos'), 'named')


def test_base():
  base = bob.bio.base.load_resource('base', 'preprocessor', preferred_package='bob.bio.face')
  assert isinstance(base, bob.bio.face.preprocessor.Base)
  assert isinstance(base, bob.bio.base.preprocessor.Preprocessor)

  # read input
  image = _image()

  preprocessed = base(image)

  assert preprocessed.ndim == 2
  assert preprocessed.dtype == numpy.float64
  assert numpy.allclose(preprocessed, bob.ip.color.rgb_to_gray(image))

  # color output
  base = bob.bio.face.preprocessor.Base(color_channel='rgb', dtype=numpy.uint8)
  colored = base(bob.ip.color.rgb_to_gray(image))

  assert colored.ndim == 3
  assert colored.dtype == numpy.uint8
  assert all(numpy.allclose(colored[c], bob.ip.color.rgb_to_gray(image)) for c in range(3))

  colored = base(image)
  assert colored.ndim == 3
  assert colored.dtype == numpy.uint8
  assert numpy.all(colored == image)






def test_face_crop():
  # read input
  image, annotation = _image(), _annotation()

  cropper = bob.bio.base.load_resource('face-crop-eyes', 'preprocessor', preferred_package='bob.bio.face')
  assert isinstance(cropper, bob.bio.face.preprocessor.FaceCrop)
  assert isinstance(cropper, bob.bio.face.preprocessor.Base)
  assert isinstance(cropper, bob.bio.base.preprocessor.Preprocessor)

  # execute face cropper
  reference = pkg_resources.resource_filename('bob.bio.face.test', 'data/cropped.hdf5')
  ref_image = _compare(cropper(image, annotation), reference, cropper.write_data, cropper.read_data)

  # test the preprocessor with fixed eye positions (which correspond to th ones
  fixed_cropper = bob.bio.face.preprocessor.FaceCrop(cropper.cropped_image_size, cropper.cropped_positions, fixed_positions = {'reye' : annotation['reye'], 'leye' : annotation['leye']})
  # result must be identical to the original face cropper (same eyes are used)
  _compare(fixed_cropper(image), reference, cropper.write_data, cropper.read_data)

  # check color cropping
  cropper.channel = 'rgb'
  cropped = cropper(image, annotation)
  assert cropped.ndim == 3
  assert cropped.shape[0] == 3
  assert cropped.shape[1:] == ref_image.shape
  assert numpy.allclose(bob.ip.color.rgb_to_gray(cropped), ref_image, atol = 1., rtol = 1.)

  # reset the configuration, so that later tests don't get screwed.
  cropper.channel = 'gray'


def test_face_detect():
  image, annotation = _image(), None

  cropper = bob.bio.base.load_resource('face-detect', 'preprocessor', preferred_package='bob.bio.face')
  assert isinstance(cropper, bob.bio.face.preprocessor.FaceDetect)
  assert isinstance(cropper, bob.bio.face.preprocessor.Base)
  assert isinstance(cropper, bob.bio.base.preprocessor.Preprocessor)
  assert cropper.flandmark is None

  # execute face detector
  reference = pkg_resources.resource_filename('bob.bio.face.test', 'data/detected.hdf5')
  _compare(cropper(image, annotation), reference, cropper.write_data, cropper.read_data)
  assert abs(cropper.quality - 33.1136586) < 1e-5

  # execute face detector with flandmark
  cropper = bob.bio.face.preprocessor.FaceDetect(face_cropper='face-crop-eyes', use_flandmark=True)
  reference = pkg_resources.resource_filename('bob.bio.face.test', 'data/flandmark.hdf5')
  _compare(cropper(image, annotation), reference, cropper.write_data, cropper.read_data)
  assert abs(cropper.quality - 33.1136586) < 1e-5

  # execute face detector with tan-triggs
  cropper = bob.bio.face.preprocessor.TanTriggs(face_cropper='landmark-detect')
  preprocessed = cropper(image, annotation)
  # load reference and perform Tan-Triggs
  detected = bob.bio.base.load(pkg_resources.resource_filename('bob.bio.face.test', 'data/flandmark.hdf5'))
  tan_triggs = bob.bio.base.load_resource('tan-triggs', 'preprocessor', preferred_package='bob.bio.face')
  reference = tan_triggs(detected)
  assert numpy.allclose(preprocessed, reference, atol=1e-5)


def test_tan_triggs():
  # read input
  image, annotation = _image(), _annotation()

  preprocessor = bob.bio.base.load_resource('tan-triggs-crop', 'preprocessor', preferred_package='bob.bio.face')
  assert isinstance(preprocessor, bob.bio.face.preprocessor.TanTriggs)
  assert isinstance(preprocessor, bob.bio.face.preprocessor.Base)
  assert isinstance(preprocessor, bob.bio.base.preprocessor.Preprocessor)
  assert isinstance(preprocessor.cropper, bob.bio.face.preprocessor.FaceCrop)

  # execute face cropper
  _compare(preprocessor(image, annotation), pkg_resources.resource_filename('bob.bio.face.test', 'data/tan_triggs_cropped.hdf5'), preprocessor.write_data, preprocessor.read_data)

  # test the preprocessor without cropping
  preprocessor = bob.bio.base.load_resource('tan-triggs', 'preprocessor', preferred_package='bob.bio.face')
  assert preprocessor.cropper is None
  # result must be identical to the original face cropper (same eyes are used)
  _compare(preprocessor(image, annotation), pkg_resources.resource_filename('bob.bio.face.test', 'data/tan_triggs_none.hdf5'), preprocessor.write_data, preprocessor.read_data)

  preprocessor = bob.bio.base.load_resource('tan-triggs-landmark', 'preprocessor', preferred_package='bob.bio.face')
  assert isinstance(preprocessor.cropper, bob.bio.face.preprocessor.FaceDetect)
  assert preprocessor.cropper.flandmark is not None


def test_inorm_lbp():
  # read input
  image, annotation = _image(), _annotation()

  preprocessor = bob.bio.base.load_resource('inorm-lbp-crop', 'preprocessor', preferred_package='bob.bio.face')
  assert isinstance(preprocessor, bob.bio.face.preprocessor.INormLBP)
  assert isinstance(preprocessor, bob.bio.face.preprocessor.Base)
  assert isinstance(preprocessor, bob.bio.base.preprocessor.Preprocessor)
  assert isinstance(preprocessor.cropper, bob.bio.face.preprocessor.FaceCrop)
  # execute preprocessor
  _compare(preprocessor(image, annotation), pkg_resources.resource_filename('bob.bio.face.test', 'data/inorm_lbp_cropped.hdf5'), preprocessor.write_data, preprocessor.read_data)

  # load the preprocessor without cropping
  preprocessor = bob.bio.base.load_resource('inorm-lbp', 'preprocessor', preferred_package='bob.bio.face')
  assert preprocessor.cropper is None
  # load the preprocessor landmark detection
  preprocessor = bob.bio.base.load_resource('inorm-lbp-landmark', 'preprocessor', preferred_package='bob.bio.face')
  assert isinstance(preprocessor.cropper, bob.bio.face.preprocessor.FaceDetect)


def test_heq():
  # read input
  image, annotation = _image(), _annotation()

  preprocessor = bob.bio.base.load_resource('histogram-crop', 'preprocessor', preferred_package='bob.bio.face')
  assert isinstance(preprocessor, bob.bio.face.preprocessor.HistogramEqualization)
  assert isinstance(preprocessor, bob.bio.face.preprocessor.Base)
  assert isinstance(preprocessor, bob.bio.base.preprocessor.Preprocessor)
  assert isinstance(preprocessor.cropper, bob.bio.face.preprocessor.FaceCrop)
  # execute preprocessor
  _compare(preprocessor(image, annotation), pkg_resources.resource_filename('bob.bio.face.test', 'data/histogram_cropped.hdf5'), preprocessor.write_data, preprocessor.read_data)

  # load the preprocessor without cropping
  preprocessor = bob.bio.base.load_resource('histogram', 'preprocessor', preferred_package='bob.bio.face')
  assert preprocessor.cropper is None
  # load the preprocessor landmark detection
  preprocessor = bob.bio.base.load_resource('histogram-landmark', 'preprocessor', preferred_package='bob.bio.face')
  assert isinstance(preprocessor.cropper, bob.bio.face.preprocessor.FaceDetect)


def test_sqi():
  # read input
  image, annotation = _image(), _annotation()

  preprocessor = bob.bio.base.load_resource('self-quotient-crop', 'preprocessor', preferred_package='bob.bio.face')
  assert isinstance(preprocessor, bob.bio.face.preprocessor.SelfQuotientImage)
  assert isinstance(preprocessor, bob.bio.face.preprocessor.Base)
  assert isinstance(preprocessor, bob.bio.base.preprocessor.Preprocessor)
  assert isinstance(preprocessor.cropper, bob.bio.face.preprocessor.FaceCrop)
  # execute preprocessor
  _compare(preprocessor(image, annotation), pkg_resources.resource_filename('bob.bio.face.test', 'data/self_quotient_cropped.hdf5'), preprocessor.write_data, preprocessor.read_data)

  # load the preprocessor without cropping
  preprocessor = bob.bio.base.load_resource('self-quotient', 'preprocessor', preferred_package='bob.bio.face')
  assert preprocessor.cropper is None
  # load the preprocessor landmark detection
  preprocessor = bob.bio.base.load_resource('self-quotient-landmark', 'preprocessor', preferred_package='bob.bio.face')
  assert isinstance(preprocessor.cropper, bob.bio.face.preprocessor.FaceDetect)
