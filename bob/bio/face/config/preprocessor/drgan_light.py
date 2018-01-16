import bob.bio.face

# This is the size of the image that this model expects
CROPPED_IMAGE_HEIGHT = 64
CROPPED_IMAGE_WIDTH = 64

RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 - 1)
LEFT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 * 3)
  
face_cropper = bob.bio.face.preprocessor.FaceCrop(
  cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
  cropped_positions = {'leye' : LEFT_EYE_POS, 'reye' : RIGHT_EYE_POS},
  color_channel='rgb'
)

preprocessor = bob.bio.face.preprocessor.FaceDetect(face_cropper=face_cropper, color_channel='rgb')


