from . import Base, bounding_box_to_annotations
from bob.ip.facedetect import bounding_box_from_annotation
from bob.ip.dlib import DlibLandmarkExtraction


class BobIpDlib(Base):
    """Annotator using bob.ip.dlib"""

    def __init__(self, **kwargs):
        super(BobIpDlib, self).__init__(**kwargs)
        self.detector = DlibLandmarkExtraction(bob_landmark_format=True)

    def annotate(self, image, **kwargs):
        landmarks = self.detector(image)
        if not landmarks:
            return {}
        bounding_box = bounding_box_from_annotation(source='eyes', **landmarks)
        landmarks.update(bounding_box_to_annotations(bounding_box))
        return landmarks
