#!/usr/bin/env python

import bob.bio.face

preprocessor = bob.bio.face.preprocessor.FaceDetect(
  face_cropper = 'face-crop-eyes'
)
