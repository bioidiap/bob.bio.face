import bob.io.base
import bob.io.base.test_utils
import bob.io.image
import bob.ip.facedetect
import numpy

face_image = bob.io.base.load(bob.io.base.test_utils.datafile(
    'testimage.jpg', 'bob.ip.facedetect'))


def test_bob_ip_facedetect():
    from bob.bio.face.annotator.bobipfacedetect import BobIpFacedetect
    annot = BobIpFacedetect()(face_image)
    assert annot['topleft'] == (110, 82), annot
    assert annot['bottomright'] == (334, 268), annot
    assert numpy.allclose(annot['quality'], 39.209601948013685), annot
