from bob.bio.base.annotator import FailSafe
from bob.bio.face.annotator import BobIpFacedetect, BobIpFlandmark

# FLandmark requires the 'topleft' and 'bottomright' annotations
annotator = FailSafe(
    [BobIpFacedetect(), BobIpFlandmark()],
    required_keys=('reye', 'leye'))
