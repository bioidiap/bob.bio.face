import bob.bio.face

preprocessor = bob.bio.face.preprocessor.SelfQuotientImage(
  face_cropper = 'face-crop-eyes'
)

preprocessor_landmark = bob.bio.face.preprocessor.SelfQuotientImage(
  face_cropper = 'landmark-detect'
)

preprocessor_no_crop = bob.bio.face.preprocessor.SelfQuotientImage(
  face_cropper = None
)
