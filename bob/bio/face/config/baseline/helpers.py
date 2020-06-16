import bob.bio.face
from sklearn.pipeline import make_pipeline
from bob.bio.base.wrappers import wrap_sample_preprocessor
from bob.pipelines import wrap
from bob.bio.face.helpers import face_crop_solver


def embedding_transformer_160x160(embedding, annotation_type, fixed_positions):
    """
    Creates a pipeline composed by and FaceCropper and an Embedding extractor.
    This transformer is suited for Facenet based architectures
    
    .. warning::
       This will resize images to :math:`160 \times 160`
    
    """

    # This is the size of the image that this model expects
    CROPPED_IMAGE_HEIGHT = 160
    CROPPED_IMAGE_WIDTH = 160
    cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)
    color_channel = "rgb"

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

    transformer = make_pipeline(
        wrap(
            ["sample"],
            face_cropper,
            transform_extra_arguments=transform_extra_arguments,
        ),
        wrap(["sample"], embedding),
    )

    return transformer


def embedding_transformer_112x112(embedding, annotation_type, fixed_positions):
    """
    Creates a pipeline composed by and FaceCropper and an Embedding extractor.
    This transformer is suited for Facenet based architectures
    
    .. warning::
       This will resize images to :math:`112 \times 112`
    
    """

    # This is the size of the image that this model expects
    CROPPED_IMAGE_HEIGHT = 112
    CROPPED_IMAGE_WIDTH = 112
    cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)
    color_channel = "rgb"

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
        RIGHT_EYE_POS = (32, 34)
        LEFT_EYE_POS = (32, 77)

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

    transformer = make_pipeline(
        wrap(
            ["sample"],
            face_cropper,
            transform_extra_arguments=transform_extra_arguments,
        ),
        wrap(["sample"], embedding),
    )

    return transformer


def crop_80x64(annotation_type, fixed_positions=None, color_channel="gray"):
    """
    Crops a face to :math:`80 \times 64`


    Parameters
    ----------

       annotation_type: str
          Type of annotations. Possible values are: `bounding-box`, `eyes-center` and None

       fixed_positions: tuple
          A tuple containing the annotations. This is used in case your input is already registered
          with fixed positions (eyes or bounding box)

       color_channel: str


    Returns
    -------

      face_cropper:
         A face cropper to be used
      
      transform_extra_arguments:
         The parameters to the transformer

    """


    # Cropping
    CROPPED_IMAGE_HEIGHT = 80
    CROPPED_IMAGE_WIDTH = CROPPED_IMAGE_HEIGHT * 4 // 5

    # eye positions for frontal images
    RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 - 1)
    LEFT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 * 3)

    cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)
    color_channel = color_channel


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

    return face_cropper, transform_extra_arguments
