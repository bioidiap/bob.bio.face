import bob.bio.face

from sklearn.pipeline import make_pipeline
from bob.bio.base.wrappers import wrap_sample_preprocessor
from bob.pipelines import wrap
from bob.bio.face.transformers import FaceNetSanderberg


# This is the size of the image that this model expects
CROPPED_IMAGE_HEIGHT = 160
CROPPED_IMAGE_WIDTH = 160


# eye positions for frontal images
RIGHT_EYE_POS = (46, 53)
LEFT_EYE_POS = (46, 107)


legacy_face_cropper = bob.bio.face.preprocessor.FaceCrop(
    cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
    cropped_positions={"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS},
    color_channel="rgb",
)

embedding = FaceNetSanderberg()

transformer = make_pipeline(
    wrap_sample_preprocessor(
        legacy_face_cropper,
        transform_extra_arguments=(("annotations", "annotations"),),
    ),
    wrap(["sample"], embedding),
)
