#!/usr/bin/env python

import bob.bio.face
import functools

# Detects the face and eye landmarks crops it using the detected eyes
preprocessor = functools.partial(bob.bio.face.preprocessor.FaceDetect,
  face_cropper = 'face-crop-eyes',
  use_flandmark = True
)

# Detects the face amd crops it without eye detection
preprocessor_no_eyes = functools.partial(bob.bio.face.preprocessor.FaceDetect,
  face_cropper = 'face-crop-eyes',
  use_flandmark = False
)
