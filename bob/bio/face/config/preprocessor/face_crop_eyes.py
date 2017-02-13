#!/usr/bin/env python

import bob.bio.face

# Cropping
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH = CROPPED_IMAGE_HEIGHT * 4 // 5

# eye positions for frontal images
RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 - 1)
LEFT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 * 3)

# define the preprocessor
preprocessor = bob.bio.face.preprocessor.FaceCrop(
  cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
  cropped_positions={'leye': LEFT_EYE_POS, 'reye': RIGHT_EYE_POS}
)

# top left and bottom right positions
TOP_LEFT_POS = (0, 0)
BOTTOM_RIGHT_POS = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)

# define the preprocessor
preprocessor_head = bob.bio.face.preprocessor.FaceCrop(
  cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
  cropped_positions={'topleft': TOP_LEFT_POS, 'bottomright': BOTTOM_RIGHT_POS}
)
