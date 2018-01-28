from .Base import Base
from .FailSafe import FailSafe


def bounding_box_to_annotations(bbx):
    landmarks = {
        'topleft': bbx.topleft,
        'bottomright': bbx.bottomright,
    }
    return landmarks
