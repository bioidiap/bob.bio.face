import bob.bio.base


def load_cropper(face_cropper):
  from .FaceCrop import FaceCrop
  from .FaceDetect import FaceDetect
  if face_cropper is None:
    cropper = None
  elif isinstance(face_cropper, str):
    cropper = bob.bio.base.load_resource(face_cropper, 'preprocessor')
  elif isinstance(face_cropper, (FaceCrop, FaceDetect)):
    cropper = face_cropper
  else:
    raise ValueError("The given face cropper type is not understood")

  assert cropper is None or isinstance(cropper,  (FaceCrop, FaceDetect))
  return cropper


def load_cropper_only(face_cropper):
  from .FaceCrop import FaceCrop
  if face_cropper is None:
    cropper = None
  elif isinstance(face_cropper, str):
    cropper = bob.bio.base.load_resource(face_cropper, 'preprocessor')
  elif isinstance(face_cropper, FaceCrop):
    cropper = face_cropper
  else:
    raise ValueError("The given face cropper type is not understood")

  assert cropper is None or isinstance(cropper, FaceCrop)
  return cropper
