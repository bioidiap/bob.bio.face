import math
import bob.io.base
import bob.ip.color
import bob.ip.facedetect
from . import Base, bounding_box_to_annotations


class BobIpFacedetect(Base):
    """Annotator using bob.ip.facedetect
    Provides topleft and bottomright annoations.

    Parameters
    ----------
    cascade : :any:`bob.ip.facedetect.Cascade`
        The file name, where a face detector cascade can be found. If ``None``,
        the default cascade for frontal faces
        :any:`bob.ip.facedetect.default_cascade` is used.

    detection_overlap : float
        See :any:`bob.ip.facedetect.detect_single_face`.

    distance : int
        See the Sampling section in the
        :ref:`Users Guide of bob.ip.facedetect <bob.ip.facedetect>`.

    scale_base : float
        See the Sampling section in the
        :ref:`Users Guide of bob.ip.facedetect <bob.ip.facedetect>`.

    lowest_scale : float
        See the Sampling section in the
        :ref:`Users Guide of bob.ip.facedetect <bob.ip.facedetect>`.

    eye_estimate : bool
        If ``True``, expected eye locations are added to the annotations.
    """

    def __init__(self, cascade=None,
                 detection_overlap=0.2, distance=2,
                 scale_base=math.pow(2., -1. / 16.), lowest_scale=0.125,
                 eye_estimate=False,
                 **kwargs):
        super(BobIpFacedetect, self).__init__(**kwargs)
        if cascade is None:
            self.cascade = bob.ip.facedetect.default_cascade()
        else:
            self.cascade = bob.ip.facedetect.Cascade(
                bob.io.base.HDF5File(cascade))
        self.detection_overlap = detection_overlap
        self.eye_estimate = eye_estimate
        self.scale_base = scale_base
        self.lowest_scale = lowest_scale
        self.distance = distance
        self.fit()

    def fit(self, X=None, y=None, **kwargs):
        self.sampler_ = bob.ip.facedetect.Sampler(
            scale_factor=self.scale_base, lowest_scale=self.lowest_scale,
            distance=self.distance)

    def annotate(self, image, **kwargs):
        """Return topleft and bottomright and expected eye positions

        Parameters
        ----------
        image : array
            Image in Bob format RGB image.
        **kwargs
            Ignored.

        Returns
        -------
        dict
            The annotations in a dictionary. The keys are topleft, bottomright,
            quality, leye, reye.
        """
        if image.ndim == 3:
            image = bob.ip.color.rgb_to_gray(image)
        bbx, quality = bob.ip.facedetect.detect_single_face(
            image, self.cascade, self.sampler_, self.detection_overlap)

        landmarks = bounding_box_to_annotations(bbx)
        landmarks['quality'] = quality
        if self.eye_estimate:
            landmarks.update(bob.ip.facedetect.expected_eye_positions(bbx))

        return landmarks
