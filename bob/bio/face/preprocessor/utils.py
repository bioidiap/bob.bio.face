import bob.bio.base
import six

import functools


def load_cropper(face_cropper):
  from .FaceCrop import FaceCrop
  from .FaceDetect import FaceDetect
  if face_cropper is None:
    cropper = None
  elif isinstance(face_cropper, six.string_types):
    cropper = bob.bio.base.load_resource(face_cropper, 'preprocessor')
  # In Dask, face_cropper is a functools. TODO: check that the object inside functool is valid
  elif isinstance(face_cropper, (FaceCrop, FaceDetect, functools.partial)):
    cropper = face_cropper
  else:
    raise ValueError("The given face cropper type is not understood")

  assert cropper is None or isinstance(cropper,  (FaceCrop, FaceDetect)) or isinstance(cropper, functools.partial)
  return cropper


def load_cropper_only(face_cropper):
  from .FaceCrop import FaceCrop
  if face_cropper is None:
    cropper = None
  elif isinstance(face_cropper, six.string_types):
    cropper = bob.bio.base.load_resource(face_cropper, 'preprocessor')
  elif isinstance(face_cropper, FaceCrop):
    cropper = face_cropper
  else:
    raise ValueError("The given face cropper type is not understood")

  assert cropper is None or isinstance(cropper, FaceCrop)
  return cropper
