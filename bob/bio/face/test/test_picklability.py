import bob.bio.face
from bob.pipelines.utils import assert_picklable


def test_face_crop():
    CROPPED_IMAGE_HEIGHT = 64
    CROPPED_IMAGE_WIDTH = 64
    RIGHT_EYE_POS = (16.0, 15.5)
    LEFT_EYE_POS = (16.0, 48.0)
    cropper = bob.bio.face.preprocessor.FaceCrop(
        cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
        cropped_positions={"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS},
        color_channel="rgb",
        dtype="uint8",
    )
    assert_picklable(cropper)
