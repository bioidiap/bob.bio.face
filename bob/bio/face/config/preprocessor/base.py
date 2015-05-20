import bob.bio.face
import numpy

preprocessor = bob.bio.face.preprocessor.Base(
  color_channel = 'gray',
  dtype = numpy.float64
)
