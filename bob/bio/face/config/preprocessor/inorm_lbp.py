import bob.bio.face
import numpy

preprocessor = bob.bio.face.preprocessor.INormLBP(
  face_cropper = 'face-crop-eyes',
  dtype = numpy.float64
)

preprocessor_landmark = bob.bio.face.preprocessor.INormLBP(
  face_cropper = 'landmark-detect',
  dtype = numpy.float64
)

preprocessor_no_crop = bob.bio.face.preprocessor.INormLBP(
  face_cropper = None,
  dtype = numpy.float64
)
