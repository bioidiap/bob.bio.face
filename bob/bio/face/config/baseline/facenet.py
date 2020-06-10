import bob.bio.face

from sklearn.pipeline import make_pipeline
from bob.bio.base.wrappers import wrap_sample_preprocessor
from bob.pipelines import wrap
from bob.bio.face.embeddings import FaceNetSanderberg
from bob.bio.face.helpers import face_crop_solver

# This is the size of the image that this model expects
CROPPED_IMAGE_HEIGHT = 160
CROPPED_IMAGE_WIDTH = 160
cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)
color_channel = "rgb"

if "database" in locals():
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
else:
    annotation_type = "eyes-center"
    fixed_positions = None

#### SOLVING THE FACE CROPPER TO BE USED
if annotation_type == "bounding-box":
    transform_extra_arguments = (("annotations", "annotations"),)
    TOP_LEFT_POS = (0, 0)
    BOTTOM_RIGHT_POS = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)

    # Detects the face and crops it without eye detection
    face_cropper = face_crop_solver(
        cropped_image_size,
        color_channel=color_channel,
        cropped_positions={"topleft": TOP_LEFT_POS, "bottomright": BOTTOM_RIGHT_POS},
        fixed_positions=fixed_positions,
    )

elif annotation_type == "eyes-center":
    transform_extra_arguments = (("annotations", "annotations"),)
    # eye positions for frontal images
    RIGHT_EYE_POS = (46, 53)
    LEFT_EYE_POS = (46, 107)

    # Detects the face and crops it without eye detection
    face_cropper = face_crop_solver(
        cropped_image_size,
        color_channel=color_channel,
        cropped_positions={"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS},
        fixed_positions=fixed_positions,
    )

else:
    transform_extra_arguments = None
    # DEFAULT TO FACE SIMPLE RESIZE
    face_cropper = face_crop_solver(cropped_image_size)


embedding = FaceNetSanderberg()

transformer = make_pipeline(
    wrap(
        ["sample"],
        face_cropper,
        transform_extra_arguments=transform_extra_arguments,
    ),
    wrap(["sample"], embedding),
)
