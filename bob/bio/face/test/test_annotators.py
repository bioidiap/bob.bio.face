import bob.io.base
import bob.io.base.test_utils
import bob.io.image
from bob.bio.face.annotator import (
    BobIpFacedetect,
    BobIpFlandmark,
    min_face_size_validator)
from bob.bio.base.annotator import FailSafe
from bob.bio.face.annotator import BobIpMTCNN
import numpy

from bob.bio.base.test.utils import is_library_available

face_image = bob.io.base.load(
    bob.io.base.test_utils.datafile(
        'testimage.jpg', 'bob.ip.facedetect'
    )
)

def _assert_mtcnn(annot):
    """
    Verifies that the MTCNN annotations are correct for ``faceimage.jpg``
    """
    assert type(annot) is dict, annot
    assert [int(x) for x in annot['topleft']] == [68, 76], annot
    assert [int(x) for x in annot['bottomright']] == [344, 274], annot
    assert [int(x) for x in annot['reye']] == [180, 129], annot
    assert [int(x) for x in annot['leye']] == [175, 220], annot
    assert numpy.allclose(annot['quality'], 0.9998975), annot

def _assert_bob_ip_facedetect(annot):
    assert annot['topleft'] == (110, 82), annot
    assert annot['bottomright'] == (334, 268), annot
    assert numpy.allclose(annot['quality'], 39.209601948013685), annot

@is_library_available("tensorflow")
def test_mtcnn_annotator():
    """
    The MTCNN annotator should return the correct annotations.
    """
    mtcnn_annotator = BobIpMTCNN()
    batch = [face_image]
    annot_batch = mtcnn_annotator(batch)
    _assert_mtcnn(annot_batch[0])

def test_bob_ip_facedetect():
    batch = [face_image]
    annot = BobIpFacedetect()(batch)
    _assert_bob_ip_facedetect(annot[0])

def test_bob_ip_facedetect_eyes():
    batch = [face_image]
    annot = BobIpFacedetect(eye_estimate=True)(batch)
    _assert_bob_ip_facedetect(annot[0])
    assert [int(x) for x in annot[0]['reye']] == [175, 128], annot
    assert [int(x) for x in annot[0]['leye']] == [175, 221], annot

def test_fail_safe():
    annotator = FailSafe(
        [BobIpFacedetect(eye_estimate=True)],
        required_keys=('reye', 'leye'),
    )
    batch = [face_image]
    annot = annotator(batch)
    _assert_bob_ip_facedetect(annot[0])
    assert [int(x) for x in annot[0]['reye']] == [175, 128], annot
    assert [int(x) for x in annot[0]['leye']] == [175, 221], annot

def test_bob_ip_flandmark():
    annotator = FailSafe(
        [BobIpFacedetect(), BobIpFlandmark()],
        required_keys=('reye', 'leye'),
    )
    batch = [face_image]
    annot = annotator(batch)
    print(annot)
    _assert_bob_ip_facedetect(annot[0])
    assert [int(x) for x in annot[0]['reye']] == [183, 127], annot
    assert [int(x) for x in annot[0]['leye']] == [174, 223], annot

def test_min_face_size_validator():
    valid = {
        'topleft': (0, 0),
        'bottomright': (32, 32),
    }
    assert min_face_size_validator(valid)

    not_valid = {
        'topleft': (0, 0),
        'bottomright': (28, 33),
    }
    assert not min_face_size_validator(not_valid)

    not_valid = {
        'topleft': (0, 0),
        'bottomright': (33, 28),
    }
    assert not min_face_size_validator(not_valid)

    assert not min_face_size_validator(None)
    assert not min_face_size_validator({})
