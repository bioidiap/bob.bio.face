from bob.bio.base.annotator import FailSafe
from bob.bio.face.annotator import BobIpFacedetect, BobIpFlandmark

annotator = FailSafe(
    [BobIpFacedetect(), BobIpFlandmark()],
    required_keys=('reye', 'leye'))
