from .Base import Base
from bob.ip.facedetect import bounding_box_from_annotation


def bounding_box_to_annotations(bbx):
    landmarks = {}
    landmarks['topleft'] = bbx.topleft_f
    landmarks['bottomright'] = bbx.bottomright_f
    return landmarks


def normalize_annotations(annotations, validator, max_age=-1):
    """Normalizes the annotations of one video sequence. It fills the
    annotations for frames from previous ones if the annotation for the current
    frame is not valid.

    Parameters
    ----------
    annotations : dict
        A dict of dict where the keys to the first dict are frame indices as
        strings (starting from 0). The inside dicts contain annotations for
        that frame.
    validator : callable
        Takes a dict (annotations) and returns True if the annotations are
        valid. This can be check based on minimal face size for example.
    max_age : :obj:`int`, optional
        An integer indicating for a how many frames a detected face is valid if
        no detection occurs after such frame. A value of -1 == forever

    Yields
    ------
    dict
        The corrected annotations of frames.
    """
    # the annotations for the current frame
    current = {}
    age = 0

    for k, annot in annotations.items():
        if validator(annot):
            current = annot
            age = 0
        elif max_age < 0 or age < max_age:
            age += 1
        else:  # no detections and age is larger than maximum allowed
            current = {}

        yield current


def min_face_size_validator(annotations, min_face_size=32):
    """Validates annotations based on face's minimal size.

    Parameters
    ----------
    annotations : dict
        The annotations in dictionary format.
    min_face_size : int, optional
        The minimal size of a face.

    Returns
    -------
    bool
        True, if the face is large enough.
    """
    bbx = bounding_box_from_annotation(**annotations)
    if bbx.size < 32:
        return False
    return True
