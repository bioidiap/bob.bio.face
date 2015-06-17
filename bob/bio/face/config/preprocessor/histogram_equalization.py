import bob.bio.face

preprocessor = bob.bio.face.preprocessor.HistogramEqualization(
  face_cropper = 'face-crop-eyes'
)

preprocessor_landmark = bob.bio.face.preprocessor.HistogramEqualization(
  face_cropper = 'landmark-detect'
)

preprocessor_no_crop = bob.bio.face.preprocessor.HistogramEqualization(
  face_cropper = None
)
