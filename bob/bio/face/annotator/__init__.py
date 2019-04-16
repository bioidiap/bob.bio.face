import bob.ip.facedetect


def bounding_box_to_annotations(bbx):
    """Converts :any:`bob.ip.facedetect.BoundingBox` to dictionary annotations.

    Parameters
    ----------
    bbx : :any:`bob.ip.facedetect.BoundingBox`
        The given bounding box.

    Returns
    -------
    dict
        A dictionary with topleft and bottomright keys.
    """
    landmarks = {
        'topleft': bbx.topleft,
        'bottomright': bbx.bottomright,
    }
    return landmarks


def min_face_size_validator(annotations, min_face_size=(32, 32)):
    """Validates annotations based on face's minimal size.

    Parameters
    ----------
    annotations : dict
        The annotations in dictionary format.
    min_face_size : (:obj:`int`, :obj:`int`), optional
        The minimal size of a face.

    Returns
    -------
    bool
        True, if the face is large enough.
    """
    if not annotations:
        return False
    for source in ('direct', 'eyes', None):
        try:
            bbx = bob.ip.facedetect.bounding_box_from_annotation(
                source=source, **annotations)
            break
        except Exception:
            if source is None:
                raise
            else:
                pass
    if bbx.size[0] < min_face_size[0] or bbx.size[1] < min_face_size[1]:
        return False
    return True


# These imports should be here to avoid circular dependencies
from .Base import Base
from .bobipfacedetect import BobIpFacedetect
from .bobipflandmark import BobIpFlandmark
from .bobipmtcnn import BobIpMTCNN


# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
    """Says object was actually declared here, and not in the import module.
    Fixing sphinx warnings of not being able to find classes, when path is
    shortened.

    Parameters
    ----------
    *args
        An iterable of objects to modify

    Resolves `Sphinx referencing issues
    <https://github.com/sphinx-doc/sphinx/issues/3048>`
    """

    for obj in args:
        obj.__module__ = __name__


__appropriate__(
    Base,
    BobIpFacedetect,
    BobIpFlandmark,
    BobIpMTCNN,
)

__all__ = [_ for _ in dir() if not _.startswith('_')]
