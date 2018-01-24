import math
from bob.io.base import HDF5File
from bob.ip.facedetect import (
    detect_single_face, Sampler, default_cascade, Cascade,
    expected_eye_positions)
from . import Base, bounding_box_to_annotations


class BobIpFacedetect(Base):
    """Annotator using bob.ip.facedetect"""

    def __init__(self, cascade=None,
                 detection_overlap=0.2, distance=2,
                 scale_base=math.pow(2., -1. / 16.), lowest_scale=0.125,
                 **kwargs):
        super(BobIpFacedetect, self).__init__(**kwargs)
        self.sampler = Sampler(
            scale_factor=scale_base, lowest_scale=lowest_scale,
            distance=distance)
        if cascade is None:
            self.cascade = default_cascade()
        else:
            self.cascade = Cascade(HDF5File(cascade))
        self.detection_overlap = detection_overlap

    def annotate(self, image, **kwargs):
        """Return topleft and bottomright and expected eye positions

        Parameters
        ----------
        image : array
            Image gray scale.
        **kwargs
            Ignored.

        Returns
        -------
        dict
            The annotations in a dictionary. The keys are topleft, bottomright,
            quality, leye, reye.
        """
        if image.ndim != 2:
            raise ValueError("The image must be gray scale (two dimensions).")
        bounding_box, quality = detect_single_face(
            image, self.cascade, self.sampler, self.detection_overlap)
        landmarks = expected_eye_positions(bounding_box)
        landmarks.update(bounding_box_to_annotations(bounding_box))
        landmarks['quality'] = quality
        return landmarks
