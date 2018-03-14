from . import Base, bounding_box_to_annotations
import bob.ip.facedetect


class BobIpDlib(Base):
    """Annotator using bob.ip.dlib"""

    def __init__(self, **kwargs):
        super(BobIpDlib, self).__init__(**kwargs)
        from bob.ip.dlib import DlibLandmarkExtraction
        self.detector = DlibLandmarkExtraction(bob_landmark_format=True)

    def annotate(self, image, **kwargs):
        """Annotates an image using bob.ip.dlib

        Parameters
        ----------
        image : numpy.array
            An RGB image in Bob format.
        **kwargs
            Ignored.

        Returns
        -------
        dict
            Annotations contain: (topleft, bottomright, leye, reye, nose,
            mouthleft, mouthright).
        """
        landmarks = self.detector(image)
        if not landmarks:
            return None
        bounding_box = bob.ip.facedetect.bounding_box_from_annotation(
            source='eyes', **landmarks)
        landmarks.update(bounding_box_to_annotations(bounding_box))
        return landmarks
