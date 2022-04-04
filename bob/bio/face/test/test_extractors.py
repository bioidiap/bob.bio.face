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

import bob.bio.base
import bob.bio.face

import numpy
import math

import bob.io.base.test_utils

import pkg_resources
from bob.db.base import read_annotation_file

import pytest
from bob.bio.base.test.utils import is_library_available

regenerate_refs = False

# Cropping
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH = CROPPED_IMAGE_HEIGHT * 4 // 5

# eye positions for frontal images
RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 - 1)
LEFT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 * 3)


def _compare(
    data,
    reference,
    write_function=bob.bio.base.save,
    read_function=bob.bio.base.load,
    atol=1e-5,
    rtol=1e-8,
):
    # write reference?
    if regenerate_refs:
        write_function(data, reference)

    # compare reference
    reference = read_function(reference)
    assert numpy.allclose(data, reference, atol=atol, rtol=rtol)


def _data():
    return bob.bio.base.load(
        pkg_resources.resource_filename("bob.bio.face.test", "data/cropped.hdf5")
    )


def test_dct_blocks():
    # read input
    data = _data()
    dct = bob.bio.face.extractor.DCTBlocks(
        block_size=12, block_overlap=11, number_of_dct_coefficients=45
    )

    assert isinstance(dct, bob.bio.face.extractor.DCTBlocks)

    # generate smaller extractor, using mixed tuple and int input for the block size and overlap
    dct = bob.bio.face.extractor.DCTBlocks(8, (0, 0), 15)

    # extract features
    feature = dct.transform([data])[0]
    assert feature.ndim == 2
    # feature dimension is one lower than the block size, since blocks are normalized by default
    assert feature.shape == (80, 14)
    reference = pkg_resources.resource_filename(
        "bob.bio.face.test", "data/dct_blocks.hdf5"
    )
    _compare(feature, reference)


def test_face_crop(height=112, width=112):
    # read input
    image, annotation = _image(), _annotation()
    CROPPED_IMAGE_HEIGHT = height
    CROPPED_IMAGE_WIDTH = width

    # preprocessor with fixed eye positions (which correspond to th ones
    fixed_cropper = bob.bio.face.preprocessor.FaceCrop(
        cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
        color_channel="rgb",
        cropped_positions={"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS},
        fixed_positions={"reye": annotation["reye"], "leye": annotation["leye"]},
    )

    cropped = fixed_cropper.transform([image])
    return cropped


def _image():
    return bob.io.base.load(
        pkg_resources.resource_filename("bob.bio.face.test", "data/testimage.jpg")
    )


def _annotation():

    return read_annotation_file(
        pkg_resources.resource_filename("bob.bio.face.test", "data/testimage.pos"),
        "named",
    )
