from . import Base


class BobIpMTCNN(Base):
    """Annotator using mtcnn in bob.ip.facedetect"""

    def __init__(self, min_size=40, factor=0.709, thresholds=(0.6, 0.7, 0.7), **kwargs):
        super(BobIpMTCNN, self).__init__(**kwargs)
        from bob.ip.facedetect.mtcnn import MTCNN

        self.detector = MTCNN(min_size=min_size, factor=factor, thresholds=thresholds)

    @property
    def min_size(self):
        return self.detector.min_size

    @property
    def factor(self):
        return self.detector.factor

    @property
    def thresholds(self):
        return self.detector.thresholds

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
        # return the annotations for the first/largest face
        annotations = self.detector.annotations(image)

        if annotations:
            return annotations[0]
        else:
            return None
