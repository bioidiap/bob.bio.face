#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from bob.bio.face.preprocessor import FaceDetect, FaceCrop, Scale
from skimage.transform import resize
import numpy as np

def face_crop_solver(
    cropped_image_size,
    color_channel="rgb",
    cropped_positions=None,
    fixed_positions=None,
    use_face_detector=False,
    dtype=np.uint8
):
    """
    Decide which face cropper to use.
    """


    if use_face_detector:
        return FaceDetect(
            face_cropper="face-crop-eyes", use_flandmark=True
        )
    else:
        # If there's not cropped positions, just resize
        if cropped_positions is None:
            return Scale(cropped_image_size)
        else:
            # Detects the face and crops it without eye detection
            return FaceCrop(
                cropped_image_size=cropped_image_size,
                cropped_positions=cropped_positions,
                color_channel=color_channel,
                fixed_positions=fixed_positions,
                dtype=dtype
            )
