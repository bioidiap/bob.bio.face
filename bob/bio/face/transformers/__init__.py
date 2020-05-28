from .facenet_sanderberg import FaceNetSanderberg
from .idiap_inception_resnet import (
    InceptionResnetv2_MsCeleb,
    InceptionResnetv2_CasiaWebFace,
    InceptionResnetv1_MsCeleb,
    InceptionResnetv1_CasiaWebFace,
)


# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
    """Says object was actually declared here, and not in the import module.
  Fixing sphinx warnings of not being able to find classes, when path is shortened.
  Parameters:

    *args: An iterable of objects to modify

  Resolves `Sphinx referencing issues
  <https://github.com/sphinx-doc/sphinx/issues/3048>`
  """

    for obj in args:
        obj.__module__ = __name__


__appropriate__(
    FaceNetSanderberg,
    InceptionResnetv2_MsCeleb,
    InceptionResnetv2_CasiaWebFace,
    InceptionResnetv1_MsCeleb,
    InceptionResnetv1_CasiaWebFace,
)
__all__ = [_ for _ in dir() if not _.startswith("_")]
