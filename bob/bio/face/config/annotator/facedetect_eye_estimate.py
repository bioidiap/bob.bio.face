from bob.bio.base.annotator import FailSafe
from bob.bio.face.annotator import BobIpFacedetect, BoundingBoxToEyes

annotator = FailSafe(
    [BobIpFacedetect(), BoundingBoxToEyes()],
    required_keys=('reye', 'leye'))
