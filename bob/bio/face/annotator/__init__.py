from .Base import Base
from .FailSafe import FailSafe


def bounding_box_to_annotations(bbx):
    landmarks = {}
    landmarks['topleft'] = bbx.topleft_f
    landmarks['bottomright'] = bbx.bottomright_f
    return landmarks
