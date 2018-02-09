from bob.ip.facedetect import bounding_box_from_annotation
from .Base import Base


def bounding_box_to_annotations(bbx):
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
    min_face_size : (int, int), optional
        The minimal size of a face.

    Returns
    -------
    bool
        True, if the face is large enough.
    """
    bbx = bounding_box_from_annotation(source='direct', **annotations)
    if bbx.size < min_face_size:
        return False
    return True


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
)

__all__ = [_ for _ in dir() if not _.startswith('_')]
