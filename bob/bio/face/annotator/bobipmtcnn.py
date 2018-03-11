from . import Base, bounding_box_to_annotations


class BobIpMTCNN(Base):
    """Annotator using bob.ip.mtcnn"""

    def __init__(self, **kwargs):
        super(BobIpMTCNN, self).__init__(**kwargs)
        from bob.ip.mtcnn import FaceDetector
        self.detector = FaceDetector()

    def annotate(self, image, **kwargs):
        bounding_box, landmarks = self.detector.detect_single_face(image)
        if not landmarks:
            return None
        landmarks.update(bounding_box_to_annotations(bounding_box))
        return landmarks
