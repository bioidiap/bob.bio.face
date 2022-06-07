import pickle

import numpy

import bob.io.base
import bob.io.base.test_utils

from bob.bio.base.annotator import FailSafe
from bob.bio.base.test.utils import is_library_available
from bob.bio.face.annotator import (
    MTCNN,
    FaceX106Landmarks,
    FaceXDetector,
    TinyFace,
    min_face_size_validator,
)

# An image with one face
face_image = bob.io.base.load(
    bob.io.base.test_utils.datafile("testimage.jpg", "bob.bio.face")
)
# An image with 6 faces
face_image_multiple = bob.io.base.load(
    bob.io.base.test_utils.datafile("test_image_multi_face.png", "bob.bio.face")
)


def _assert_mtcnn(annot):
    """
    Verifies that the MTCNN annotations are correct for ``faceimage.jpg``
    """
    assert type(annot) is dict, annot
    assert [int(x) for x in annot["topleft"]] == [68, 76], annot
    assert [int(x) for x in annot["bottomright"]] == [344, 274], annot
    assert [int(x) for x in annot["reye"]] == [180, 129], annot
    assert [int(x) for x in annot["leye"]] == [175, 220], annot
    assert numpy.allclose(annot["quality"], 0.9998975), annot


def _assert_tinyface(annot):
    """
    Verifies that the Tinyface annotations are correct for ``faceimage.jpg``
    """

    assert type(annot) is dict, annot
    assert [int(x) for x in annot["topleft"]] == [59, 57], annot
    assert [int(x) for x in annot["bottomright"]] == [338, 284], annot
    assert [int(x) for x in annot["reye"]] == [162, 125], annot
    assert [int(x) for x in annot["leye"]] == [162, 216], annot


def _assert_annotator_is_serializable(annotator):
    """
    Verifies that a given `annotator` can be serialized / deserialized properly.
    This is required to be usable on the grid with dask.
    """
    serializable = True
    try:
        pickle.loads(pickle.dumps(annotator))
    except Exception:
        serializable = False
    assert serializable, "Can not serialize {}".format(annotator)


@is_library_available("tensorflow")
def test_mtcnn_annotator():
    """
    The MTCNN annotator should return the correct annotations.
    """
    mtcnn_annotator = MTCNN()
    batch = [face_image]
    annot_batch = mtcnn_annotator(batch)
    _assert_mtcnn(annot_batch[0])

    annot = mtcnn_annotator.annotations(face_image_multiple)
    assert len(annot) == 6

    _assert_annotator_is_serializable(mtcnn_annotator)


@is_library_available("cv2")
def test_faceX106_annotator():
    """
    FaceX-Zoo annotator
    """
    faceX_annotator = FaceX106Landmarks()
    batch = [face_image]
    annot_batch = faceX_annotator(batch)[0]
    assert annot_batch.shape == (106, 2)


@is_library_available("cv2")
def test_faceX_detector():
    """
    FaceX-Zoo annotator
    """
    faceX_annotator = FaceXDetector()
    batch = [face_image]
    annot_batch = faceX_annotator(batch)
    assert annot_batch[0].shape == (5,)


@is_library_available("mxnet")
def test_tinyface_annotator():
    """
    The Tiny face annotator should return the correct annotations.
    """
    tinyface_annotator = TinyFace()
    batch = [face_image]
    annot_batch = tinyface_annotator(batch)
    _assert_tinyface(annot_batch[0])

    annot = tinyface_annotator.annotations(face_image_multiple)
    assert len(annot) == 6


@is_library_available("tensorflow")
def test_fail_safe():
    annotator = FailSafe(
        [MTCNN()],
        required_keys=("reye", "leye"),
    )
    batch = [face_image]
    annot = annotator(batch)
    _assert_mtcnn(annot[0])


def test_min_face_size_validator():
    valid = {
        "topleft": (0, 0),
        "bottomright": (32, 32),
    }
    assert min_face_size_validator(valid)

    not_valid = {
        "topleft": (0, 0),
        "bottomright": (28, 33),
    }
    assert not min_face_size_validator(not_valid)

    not_valid = {
        "topleft": (0, 0),
        "bottomright": (33, 28),
    }
    assert not min_face_size_validator(not_valid)

    assert not min_face_size_validator(None)
    assert not min_face_size_validator({})
