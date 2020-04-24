#!/usr/bin/env python

import bob.bio.face

# Detects the face and eye landmarks crops it using the detected eyes
preprocessor = bob.bio.face.preprocessor.FaceDetect(
  face_cropper = 'face-crop-eyes',
  use_flandmark = True
)

# Detects the face amd crops it without eye detection
preprocessor_no_eyes = bob.bio.face.preprocessor.FaceDetect(
  face_cropper = 'face-crop-eyes',
  use_flandmark = False
)
