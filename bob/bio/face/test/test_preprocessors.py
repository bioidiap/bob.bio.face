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


import numpy

import pkg_resources

regenerate_refs = False

import bob.bio.base
import bob.bio.face
import bob.db.base


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
    return reference


def _image():
    return bob.io.base.load(
        pkg_resources.resource_filename("bob.bio.face.test", "data/testimage.jpg")
    )


def _annotation():
    return bob.db.base.read_annotation_file(
        pkg_resources.resource_filename("bob.bio.face.test", "data/testimage.pos"),
        "named",
    )


def test_base():
    base = bob.bio.face.preprocessor.Base(
      color_channel = 'gray',
      dtype = numpy.float64
    )
    assert isinstance(base, bob.bio.face.preprocessor.Base)

    # read input
    image = _image()

    preprocessed = base.transform([image])[0]

    assert preprocessed.ndim == 2
    assert preprocessed.dtype == numpy.float64
    assert numpy.allclose(preprocessed, bob.ip.color.rgb_to_gray(image))

    # color output
    base = bob.bio.face.preprocessor.Base(color_channel="rgb", dtype=numpy.uint8)
    colored = base.transform([bob.ip.color.rgb_to_gray(image)])[0]

    assert colored.ndim == 3
    assert colored.dtype == numpy.uint8
    assert all(
        numpy.allclose(colored[c], bob.ip.color.rgb_to_gray(image)) for c in range(3)
    )

    colored = base.transform([image])[0]
    assert colored.ndim == 3
    assert colored.dtype == numpy.uint8
    assert numpy.all(colored == image)


def test_face_crop():
    # read input
    image, annotation = _image(), _annotation()

    # define the preprocessor
    cropper = bob.bio.face.preprocessor.FaceCrop(
      cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
      cropped_positions={'leye': LEFT_EYE_POS, 'reye': RIGHT_EYE_POS}
    )

    assert isinstance(cropper, bob.bio.face.preprocessor.FaceCrop)
    assert isinstance(cropper, bob.bio.face.preprocessor.Base)

    # execute face cropper
    reference = pkg_resources.resource_filename(
        "bob.bio.face.test", "data/cropped.hdf5"
    )
    ref_image = _compare(cropper.transform([image], [annotation]), reference)


    # test the preprocessor with fixed eye positions (which correspond to th ones
    fixed_cropper = bob.bio.face.preprocessor.FaceCrop(
        cropper.cropped_image_size,
        cropper.cropped_positions,
        fixed_positions={"reye": annotation["reye"], "leye": annotation["leye"]},
    )
    # result must be identical to the original face cropper (same eyes are used)
    _compare(fixed_cropper.transform([image]), reference)

    # check color cropping
    cropper.color_channel = "rgb"
    cropped = cropper.transform([image], [annotation])[0]
    assert cropped.ndim == 3
    assert cropped.shape[0] == 3
    assert cropped.shape[1:] == ref_image.shape
    assert numpy.allclose(
        bob.ip.color.rgb_to_gray(cropped), ref_image, atol=1.0, rtol=1.0
    )

    # test a ValueError is raised if eye annotations are swapped
    try:
        annot = dict(reye=annotation["leye"], leye=annotation["reye"])
        cropper.transform([image], [annot])
        assert 0, "FaceCrop did not raise a ValueError for swapped eye annotations"
    except ValueError:
        pass

    # reset the configuration, so that later tests don't get screwed.
    cropper.color_channel = "gray"

def test_multi_face_crop():
    # read input
    image = _image()
    eye_annotation, bbox_annotation = [
            bob.db.base.read_annotation_file(
                pkg_resources.resource_filename("bob.bio.face.test", "data/" + filename + ".pos"),
                "named"
            )
            for filename in ["testimage", "testimage_bbox"]
        ]

    # define the preprocessor
    cropper = bob.bio.face.preprocessor.MultiFaceCrop(
        cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
        cropped_positions_list=[
            {'leye': LEFT_EYE_POS, 'reye': RIGHT_EYE_POS},
            {'topleft': (0, 0), 'bottomright': (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)}
        ]
    )

    # execute face cropper
    eye_reference, bbox_reference = [
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/" + filename + ".hdf5"
        )
        for filename in ["cropped", "cropped_bbox"]
    ]

    eye_cropped, bbox_cropped = cropper.transform([image, image], [eye_annotation, bbox_annotation])
    
    # Compare the cropped results to the reference
    _compare(eye_cropped, eye_reference)
    _compare(bbox_cropped, bbox_reference)

     # test a ValueError is raised if the annotations don't match any cropper
    try:
        annot = dict(landmark_A=(60, 60), landmark_B=(120, 120))
        cropper.transform([image], [annot])
        assert 0, "MultiFaceCrop did not raise a ValueError for annotations matching no cropper"
    except ValueError:
        pass

    # test a ValueError is raised if the annotations match several croppers
    try:
        annot = {**eye_annotation, **bbox_annotation}
        cropper.transform([image], [annot])
        assert 0, "MultiFaceCrop did not raise a ValueError for annotations matching several croppers"
    except ValueError:
        pass


def test_tan_triggs():
    # read input
    image, annotation = _image(), _annotation()

    face_cropper = bob.bio.face.preprocessor.FaceCrop(
      cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
      cropped_positions={'leye': LEFT_EYE_POS, 'reye': RIGHT_EYE_POS}
    )

    preprocessor = bob.bio.face.preprocessor.TanTriggs(face_cropper=face_cropper)

    assert isinstance(preprocessor, bob.bio.face.preprocessor.TanTriggs)
    assert isinstance(preprocessor, bob.bio.face.preprocessor.Base)
    assert isinstance(preprocessor.cropper, bob.bio.face.preprocessor.FaceCrop)

    # execute face cropper
    _compare(
        preprocessor.transform([image], [annotation]),
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/tan_triggs_cropped.hdf5"
        )
    )

    # test the preprocessor without cropping
    preprocessor = bob.bio.face.preprocessor.TanTriggs(face_cropper=None)
    assert preprocessor.cropper is None
    # result must be identical to the original face cropper (same eyes are used)
    _compare(
        preprocessor.transform([image], [annotation]),
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/tan_triggs_none.hdf5"
        )
    )


def test_inorm_lbp():
    # read input
    image, annotation = _image(), _annotation()
    face_cropper = bob.bio.face.preprocessor.FaceCrop(
      cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
      cropped_positions={'leye': LEFT_EYE_POS, 'reye': RIGHT_EYE_POS}
    )

    preprocessor = bob.bio.face.preprocessor.INormLBP(
      face_cropper = face_cropper,
      dtype = numpy.float64
    )

    assert isinstance(preprocessor, bob.bio.face.preprocessor.INormLBP)
    assert isinstance(preprocessor, bob.bio.face.preprocessor.Base)
    assert isinstance(preprocessor.cropper, bob.bio.face.preprocessor.FaceCrop)
    # execute preprocessor
    _compare(
        preprocessor.transform([image], [annotation]),
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/inorm_lbp_cropped.hdf5"
        )
    )

    # load the preprocessor without cropping
    preprocessor = bob.bio.face.preprocessor.INormLBP(
      face_cropper = None,
    )
    assert preprocessor.cropper is None


def test_heq():
    # read input
    image, annotation = _image(), _annotation()

    face_cropper = bob.bio.face.preprocessor.FaceCrop(
      cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
      cropped_positions={'leye': LEFT_EYE_POS, 'reye': RIGHT_EYE_POS}
    )
    preprocessor = bob.bio.face.preprocessor.HistogramEqualization(
      face_cropper = face_cropper
    )

    assert isinstance(preprocessor, bob.bio.face.preprocessor.HistogramEqualization)
    assert isinstance(preprocessor, bob.bio.face.preprocessor.Base)
    assert isinstance(preprocessor.cropper, bob.bio.face.preprocessor.FaceCrop)
    # execute preprocessor
    _compare(
        preprocessor.transform([image],[annotation]),
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/histogram_cropped.hdf5"
        ),
    )

    # load the preprocessor without cropping
    preprocessor = bob.bio.face.preprocessor.HistogramEqualization(
      face_cropper = None
    )
    assert preprocessor.cropper is None
    # load the preprocessor landmark detection


def test_sqi():
    # read input
    image, annotation = _image(), _annotation()

    face_cropper = bob.bio.face.preprocessor.FaceCrop(
      cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
      cropped_positions={'leye': LEFT_EYE_POS, 'reye': RIGHT_EYE_POS}
    )
    preprocessor = bob.bio.face.preprocessor.SelfQuotientImage(
      face_cropper = face_cropper
    )

    assert isinstance(preprocessor, bob.bio.face.preprocessor.SelfQuotientImage)
    assert isinstance(preprocessor, bob.bio.face.preprocessor.Base)
    assert isinstance(preprocessor.cropper, bob.bio.face.preprocessor.FaceCrop)
    # execute preprocessor
    _compare(
        preprocessor.transform([image], [annotation]),
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/self_quotient_cropped.hdf5"
        )
    )

    # load the preprocessor without cropping
    preprocessor = bob.bio.face.preprocessor.SelfQuotientImage(
      face_cropper = None
    )
    assert preprocessor.cropper is None
