import bob.io.base
import bob.io.base.test_utils
import bob.io.image
from bob.bio.face.annotator import (
    BobIpFacedetect, BoundingBoxToEyes, BobIpFlandmark)
from bob.bio.base.annotator import FailSafe
import numpy

face_image = bob.io.base.load(bob.io.base.test_utils.datafile(
    'testimage.jpg', 'bob.ip.facedetect'))


def _assert_bob_ip_facedetect(annot):
    assert annot['topleft'] == (110, 82), annot
    assert annot['bottomright'] == (334, 268), annot
    assert numpy.allclose(annot['quality'], 39.209601948013685), annot


def test_bob_ip_facedetect():
    from bob.bio.face.annotator.bobipfacedetect import BobIpFacedetect
    annot = BobIpFacedetect()(face_image)
    _assert_bob_ip_facedetect(annot)


def test_bob_ip_facedetect_eyes():
    annotator = FailSafe(
        [BobIpFacedetect(), BoundingBoxToEyes()],
        required_keys=('reye', 'leye'),
    )

    annot = annotator(face_image)

    _assert_bob_ip_facedetect(annot)
    assert [int(x) for x in annot['reye']] == [175, 128], annot
    assert [int(x) for x in annot['leye']] == [175, 221], annot


def test_bob_ip_flandmark():
    annotator = FailSafe(
        [BobIpFacedetect(), BobIpFlandmark()],
        required_keys=('reye', 'leye'),
    )

    annot = annotator(face_image)

    _assert_bob_ip_facedetect(annot)
    assert [int(x) for x in annot['reye']] == [183, 127], annot
    assert [int(x) for x in annot['leye']] == [174, 223], annot
