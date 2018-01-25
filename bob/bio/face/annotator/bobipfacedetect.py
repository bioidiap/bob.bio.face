import math
from bob.io.base import HDF5File
from bob.ip.color import rgb_to_gray
from bob.ip.facedetect import (
    detect_single_face, Sampler, default_cascade, Cascade,
    bounding_box_from_annotation, expected_eye_positions)
from . import Base, bounding_box_to_annotations


class BobIpFacedetect(Base):
    """Annotator using bob.ip.facedetect
    Provides topleft and bottomright annoations.
    """

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
            Image is Bob format RGB image.
        **kwargs
            Ignored.

        Returns
        -------
        dict
            The annotations in a dictionary. The keys are topleft, bottomright,
            quality, leye, reye.
        """
        image = rgb_to_gray(image)
        bounding_box, quality = detect_single_face(
            image, self.cascade, self.sampler, self.detection_overlap)
        landmarks = bounding_box_to_annotations(bounding_box)
        landmarks['quality'] = quality
        return landmarks


class BoundingBoxToEyes(Base):
    """Converts bounding box annotations to eye locations. The bounding box's
    annotations is expected to have come from :any:`BobIpFacedetect`.
    """

    def annotate(self, image, annotations, **kwargs):
        bbx = bounding_box_from_annotation(source='direct', **annotations)
        annotations = dict(annotations)
        annotations.update(expected_eye_positions(bbx))
        return annotations
