import bob.ip.facedetect.tinyface
from . import Base
import cv2 as cv


class BobIpTinyface(Base):
    """Annotator using tinyface in bob.ip.facedetect"""

    def __init__(self, **kwargs):
        super(BobIpTinyface, self).__init__(**kwargs)
        self.tinyface = bob.ip.facedetect.tinyface.TinyFacesDetector(prob_thresh=0.5)

    def annotate(self, image, **kwargs):
        """Annotates an image using tinyface

        Parameters
        ----------
        image : numpy.array
            An RGB image in Bob format.
        **kwargs
            Ignored.

        Returns
        -------
        dict
            Annotations with (topleft, bottomright) keys (or None).
        """

        if annotations is not None:
            return annotations[0]
        else:
            return None
