import bob.bio.face
from sklearn.pipeline import make_pipeline
from bob.bio.base.wrappers import wrap_sample_preprocessor
from bob.pipelines import wrap
from bob.bio.face.helpers import face_crop_solver
import numpy as np
import logging

logger = logging.getLogger(__name__)


def embedding_transformer_default_cropping(cropped_image_size, annotation_type):
    """
    Computes the default cropped positions for the FaceCropper used with Facenet-like 
    Embedding extractors, proportionally to the target image size


    Parameters
    ----------
       cropped_image_size : tuple
          A tuple (HEIGHT, WIDTH) describing the target size of the cropped image.

       annotation_type: str or list of str
          Type of annotations. Possible values are: `bounding-box`, `eyes-center`, 'left-profile', 
          'right-profile'  and None, or a combination of those as a list

    Returns
    -------

      cropped_positions:
         The dictionary of cropped positions that will be feeded to the FaceCropper, or a list of such dictionaries if
         ``annotation_type`` is a list
    """
    if isinstance(annotation_type, list):
        return [
            embedding_transformer_default_cropping(cropped_image_size, item)
            for item in annotation_type
        ]

    CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH = cropped_image_size

    if annotation_type == "bounding-box":

        TOP_LEFT_POS = (0, 0)
        BOTTOM_RIGHT_POS = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)
        cropped_positions = {"topleft": TOP_LEFT_POS, "bottomright": BOTTOM_RIGHT_POS}

    elif annotation_type == "eyes-center":

        RIGHT_EYE_POS = (
            round(2 / 7 * CROPPED_IMAGE_HEIGHT),
            round(1 / 3 * CROPPED_IMAGE_WIDTH),
        )
        LEFT_EYE_POS = (
            round(2 / 7 * CROPPED_IMAGE_HEIGHT),
            round(2 / 3 * CROPPED_IMAGE_WIDTH),
        )
        cropped_positions = {"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS}

    elif annotation_type == "left-profile":

        EYE_POS = (
            round(2 / 7 * CROPPED_IMAGE_HEIGHT),
            round(3 / 8 * CROPPED_IMAGE_WIDTH),
        )
        MOUTH_POS = (
            round(5 / 7 * CROPPED_IMAGE_HEIGHT),
            round(3 / 8 * CROPPED_IMAGE_WIDTH),
        )
        cropped_positions = {"leye": EYE_POS, "mouth": MOUTH_POS}

    elif annotation_type == "right-profile":

        EYE_POS = (
            round(2 / 7 * CROPPED_IMAGE_HEIGHT),
            round(5 / 8 * CROPPED_IMAGE_WIDTH),
        )
        MOUTH_POS = (
            round(5 / 7 * CROPPED_IMAGE_HEIGHT),
            round(5 / 8 * CROPPED_IMAGE_WIDTH),
        )
        cropped_positions = {"reye": EYE_POS, "mouth": MOUTH_POS}

    else:

        logger.warning(
            f"Annotation type {annotation_type} is not supported. Input images will be fully scaled."
        )
        cropped_positions = None

    return cropped_positions


def legacy_default_cropping(cropped_image_size, annotation_type):
    """
    Computes the default cropped positions for the FaceCropper used with legacy extractors, 
    proportionally to the target image size


    Parameters
    ----------
       cropped_image_size : tuple
          A tuple (HEIGHT, WIDTH) describing the target size of the cropped image.

       annotation_type: str
          Type of annotations. Possible values are: `bounding-box`, `eyes-center`, 'left-profile', 
          'right-profile' and None, or a combination of those as a list

    Returns
    -------

      cropped_positions:
         The dictionary of cropped positions that will be feeded to the FaceCropper, or a list of such dictionaries if
         ``annotation_type`` is a list
    """
    if isinstance(annotation_type, list):
        return [
            legacy_default_cropping(cropped_image_size, item)
            for item in annotation_type
        ]

    CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH = cropped_image_size

    if annotation_type == "bounding-box":

        TOP_LEFT_POS = (0, 0)
        BOTTOM_RIGHT_POS = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)
        cropped_positions = {"topleft": TOP_LEFT_POS, "bottomright": BOTTOM_RIGHT_POS}

    elif annotation_type == "eyes-center":

        RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 - 1)
        LEFT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 * 3)
        cropped_positions = {"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS}

    elif annotation_type == "left-profile":
        # Main reference https://gitlab.idiap.ch/bob/bob.chapter.FRICE/-/blob/master/bob/chapter/FRICE/script/pose.py
        EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 7 * 3 - 2)
        MOUTH_POS = (CROPPED_IMAGE_HEIGHT // 3 * 2, CROPPED_IMAGE_WIDTH // 7 * 3 - 2)
        cropped_positions = {"leye": EYE_POS, "mouth": MOUTH_POS}

    elif annotation_type == "right-profile":
        # Main reference https://gitlab.idiap.ch/bob/bob.chapter.FRICE/-/blob/master/bob/chapter/FRICE/script/pose.py
        EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 7 * 4 + 2)
        MOUTH_POS = (CROPPED_IMAGE_HEIGHT // 3 * 2, CROPPED_IMAGE_WIDTH // 7 * 4 + 2)
        cropped_positions = {"reye": EYE_POS, "mouth": MOUTH_POS}

    else:

        logger.warning(
            f"Annotation type {annotation_type} is not supported. Input images will be fully scaled."
        )
        cropped_positions = None

    return cropped_positions


def embedding_transformer(
    cropped_image_size,
    embedding,
    annotation_type,
    cropped_positions,
    fixed_positions=None,
    color_channel="rgb",
):
    """
    Creates a pipeline composed by and FaceCropper and an Embedding extractor.
    This transformer is suited for Facenet based architectures
    
    .. warning::
       This will resize images to the requested `image_size`
    
    """
    face_cropper = face_crop_solver(
        cropped_image_size,
        color_channel=color_channel,
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        dtype="float64",
    )

    transform_extra_arguments = (
        None
        if (cropped_positions is None or fixed_positions is not None)
        else (("annotations", "annotations"),)
    )

    transformer = make_pipeline(
        wrap(
            ["sample"],
            face_cropper,
            transform_extra_arguments=transform_extra_arguments,
        ),
        wrap(["sample"], embedding),
    )

    return transformer


def embedding_transformer_160x160(
    embedding, annotation_type, fixed_positions, color_channel="rgb"
):
    """
    Creates a pipeline composed by and FaceCropper and an Embedding extractor.
    This transformer is suited for Facenet based architectures
    
    .. warning::
       This will resize images to :math:`160 \times 160`
    
    """
    cropped_positions = embedding_transformer_default_cropping(
        (160, 160), annotation_type
    )

    return embedding_transformer(
        (160, 160),
        embedding,
        annotation_type,
        cropped_positions,
        fixed_positions,
        color_channel=color_channel,
    )


def embedding_transformer_112x112(
    embedding, annotation_type, fixed_positions, color_channel="rgb"
):
    """
    Creates a pipeline composed by and FaceCropper and an Embedding extractor.
    This transformer is suited for Facenet based architectures
    
    .. warning::
       This will resize images to :math:`112 \times 112`
    
    """
    cropped_image_size = (112, 112)
    if annotation_type == "eyes-center":
        # Hard coding eye positions for backward consistency
        cropped_positions = {
            "leye": (55, 81),
            "reye": (55, 42),
        }

    else:
        # Will use default
        cropped_positions = embedding_transformer_default_cropping(
            cropped_image_size, annotation_type
        )

    return embedding_transformer(
        cropped_image_size,
        embedding,
        annotation_type,
        cropped_positions,
        fixed_positions,
        color_channel=color_channel,
    )


def embedding_transformer_224x224(
    embedding, annotation_type, fixed_positions, color_channel="rgb"
):
    """
    Creates a pipeline composed by and FaceCropper and an Embedding extractor.
    This transformer is suited for Facenet based architectures
    
    .. warning::
       This will resize images to :math:`112 \times 112`
    
    """
    cropped_image_size = (224, 224)
    if annotation_type == "eyes-center":
        # Hard coding eye positions for backward consistency
        cropped_positions = {"leye": (65, 150), "reye": (65, 77)}
    else:
        # Will use default
        cropped_positions = embedding_transformer_default_cropping(
            cropped_image_size, annotation_type
        )

    return embedding_transformer(
        cropped_image_size,
        embedding,
        annotation_type,
        cropped_positions,
        fixed_positions,
        color_channel=color_channel,
    )


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
    color_channel = color_channel
    dtype = np.float64

    # Cropping
    CROPPED_IMAGE_HEIGHT = 80
    CROPPED_IMAGE_WIDTH = CROPPED_IMAGE_HEIGHT * 4 // 5
    cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)

    cropped_positions = legacy_default_cropping(cropped_image_size, annotation_type)

    face_cropper = face_crop_solver(
        cropped_image_size,
        color_channel=color_channel,
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        dtype=dtype,
    )

    transform_extra_arguments = (
        None
        if (cropped_positions is None or fixed_positions is not None)
        else (("annotations", "annotations"),)
    )

    return face_cropper, transform_extra_arguments
