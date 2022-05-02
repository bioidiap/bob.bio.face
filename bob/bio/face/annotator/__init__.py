# isort: skip_file
from .utils import *  # noqa: F401,F403
from .Base import Base
from .mtcnn import MTCNN
from .tinyface import TinyFace
from .faceX_106landmarks import FaceX106Landmarks, FaceXDetector
from .utils import BoundingBox

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
    MTCNN,
    TinyFace,
    FaceX106Landmarks,
    FaceXDetector,
    BoundingBox,
)

__all__ = [_ for _ in dir() if not _.startswith("_")]
