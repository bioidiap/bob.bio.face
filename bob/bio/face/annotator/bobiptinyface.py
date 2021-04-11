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

        annotations = self.tinyface.detect(image)

        if annotations is not None:
            r = annotations[0]
            return {"topleft": (r[0], r[1]), "bottomright": (r[2], r[3])}
        else:
            return None
