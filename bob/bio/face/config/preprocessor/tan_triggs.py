import bob.bio.face

preprocessor = bob.bio.face.preprocessor.TanTriggs(
  face_cropper = 'face-crop-eyes'
)

preprocessor_no_crop = bob.bio.face.preprocessor.TanTriggs(
  face_cropper = None
)
