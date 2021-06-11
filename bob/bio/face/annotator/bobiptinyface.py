import bob.ip.facedetect.tinyface
from . import Base


class BobIpTinyface(Base):
    """Annotator using tinyface in bob.ip.facedetect"""

    def __init__(self, **kwargs):
        super(BobIpTinyface, self).__init__(**kwargs)
        self.detector = bob.ip.facedetect.tinyface.TinyFacesDetector(prob_thresh=0.5)

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

        # return the annotations for the first/largest face
        annotations = self.detector.detect(image)

        if annotations is not None:
            return annotations[0]
        else:
            return None
