from . import Base


class BobIpMTCNN(Base):
    """Annotator using mtcnn in bob.ip.tensorflow_extractor"""

    def __init__(self, **kwargs):
        super(BobIpMTCNN, self).__init__(**kwargs)
        from bob.ip.tensorflow_extractor import MTCNN
        self.detector = MTCNN()

    def annotate(self, image, **kwargs):
        """Annotates an image using mtcnn

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
            mouthleft, mouthright, quality).
        """
        # return the annotations for the first/largest face.
        return self.detector.annotations(image)[0]
