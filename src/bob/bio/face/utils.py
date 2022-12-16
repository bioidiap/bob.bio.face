import logging

from sklearn.pipeline import Pipeline

from bob.pipelines import wrap

from .preprocessor import (
    BoundingBoxAnnotatorCrop,
    FaceCrop,
    MultiFaceCrop,
    Scale,
)
from .preprocessor.croppers import FaceEyesNorm

logger = logging.getLogger(__name__)


def lookup_config_from_database(database):
    """
    Read configuration values that might be already defined in the database configuration
    file.
    """
    if database is not None:
        annotation_type = database.annotation_type
        fixed_positions = database.fixed_positions
        memory_demanding = (
            database.memory_demanding
            if hasattr(database, "memory_demanding")
            else False
        )

    else:
        annotation_type = None
        fixed_positions = None
        memory_demanding = False

    return annotation_type, fixed_positions, memory_demanding


def cropped_positions_arcface(annotation_type="eyes-center"):
    """
    Returns the 112 x 112 crop used in iResnet based models
    The crop follows the following rule:

        - In X --> (112/2)-1
        - In Y, leye --> 16+(112/2) --> 72
        - In Y, reye --> (112/2)-16 --> 40

    This will leave 16 pixels between left eye and left border and right eye and right border

    For reference, https://github.com/deepinsight/insightface/blob/master/recognition/arcface_mxnet/common/face_align.py
    contains the cropping code for training the original ArcFace-InsightFace model. Due to this code not being very explicit,
    we choose to pick our own default cropped positions. They have been tested to provide good evaluation performance
    on the Mobio dataset.

    For sensitive applications, you can use custom cropped position that you optimize for your specific dataset,
    such as is done in https://gitlab.idiap.ch/bob/bob.bio.face/-/blob/master/notebooks/50-shades-of-face.ipynb

    """

    if isinstance(annotation_type, list):
        return [cropped_positions_arcface(item) for item in annotation_type]

    if annotation_type == "eyes-center" or annotation_type == "bounding-box":
        cropped_positions = {
            "leye": (55, 72),
            "reye": (55, 40),
        }
    elif annotation_type == "left-profile":

        cropped_positions = {"leye": (52, 56), "mouth": (91, 56)}
    elif annotation_type == "right-profile":
        return {"reye": (52, 56), "mouth": (91, 56)}
    else:
        raise ValueError(
            f"Annotations of the type `{annotation_type}` not supported"
        )

    return cropped_positions


def dnn_default_cropping(cropped_image_size, annotation_type):
    """
    Computes the default cropped positions for the FaceCropper used with Neural-Net based
    extractors, proportionally to the target image size


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
            dnn_default_cropping(cropped_image_size, item)
            for item in annotation_type
        ]

    CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH = cropped_image_size

    cropped_positions = {}

    if annotation_type == "bounding-box":

        TOP_LEFT_POS = (0, 0)
        BOTTOM_RIGHT_POS = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)
        cropped_positions.update(
            {"topleft": TOP_LEFT_POS, "bottomright": BOTTOM_RIGHT_POS}
        )

    if annotation_type in ["bounding-box", "eyes-center"]:
        # We also add cropped eye positions if `bounding-box`, to work with the BoundingBoxCropAnnotator
        RIGHT_EYE_POS = (
            round(2 / 7 * CROPPED_IMAGE_HEIGHT),
            round(1 / 3 * CROPPED_IMAGE_WIDTH),
        )
        LEFT_EYE_POS = (
            round(2 / 7 * CROPPED_IMAGE_HEIGHT),
            round(2 / 3 * CROPPED_IMAGE_WIDTH),
        )
        cropped_positions.update({"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS})

    elif annotation_type == "left-profile":

        EYE_POS = (
            round(2 / 7 * CROPPED_IMAGE_HEIGHT),
            round(3 / 8 * CROPPED_IMAGE_WIDTH),
        )
        MOUTH_POS = (
            round(5 / 7 * CROPPED_IMAGE_HEIGHT),
            round(3 / 8 * CROPPED_IMAGE_WIDTH),
        )
        cropped_positions.update({"leye": EYE_POS, "mouth": MOUTH_POS})

    elif annotation_type == "right-profile":

        EYE_POS = (
            round(2 / 7 * CROPPED_IMAGE_HEIGHT),
            round(5 / 8 * CROPPED_IMAGE_WIDTH),
        )
        MOUTH_POS = (
            round(5 / 7 * CROPPED_IMAGE_HEIGHT),
            round(5 / 8 * CROPPED_IMAGE_WIDTH),
        )
        cropped_positions.update({"reye": EYE_POS, "mouth": MOUTH_POS})

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

    cropped_positions = {}

    if annotation_type == "bounding-box":

        TOP_LEFT_POS = (0, 0)
        BOTTOM_RIGHT_POS = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)
        cropped_positions.update(
            {"topleft": TOP_LEFT_POS, "bottomright": BOTTOM_RIGHT_POS}
        )

    if annotation_type in ["bounding-box", "eyes-center"]:
        # We also add cropped eye positions if `bounding-box`, to work with the BoundingBoxCropAnnotator

        RIGHT_EYE_POS = (
            CROPPED_IMAGE_HEIGHT // 5,
            CROPPED_IMAGE_WIDTH // 4 - 1,
        )
        LEFT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 * 3)
        cropped_positions.update({"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS})

    elif annotation_type == "left-profile":
        # Main reference https://gitlab.idiap.ch/bob/bob.chapter.FRICE/-/blob/master/bob/chapter/FRICE/script/pose.py
        EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 7 * 3 - 2)
        MOUTH_POS = (
            CROPPED_IMAGE_HEIGHT // 3 * 2,
            CROPPED_IMAGE_WIDTH // 7 * 3 - 2,
        )
        cropped_positions.update({"leye": EYE_POS, "mouth": MOUTH_POS})

    elif annotation_type == "right-profile":
        # Main reference https://gitlab.idiap.ch/bob/bob.chapter.FRICE/-/blob/master/bob/chapter/FRICE/script/pose.py
        EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 7 * 4 + 2)
        MOUTH_POS = (
            CROPPED_IMAGE_HEIGHT // 3 * 2,
            CROPPED_IMAGE_WIDTH // 7 * 4 + 2,
        )
        cropped_positions.update({"reye": EYE_POS, "mouth": MOUTH_POS})

    else:

        logger.warning(
            f"Annotation type {annotation_type} is not supported. Input images will be fully scaled."
        )
        cropped_positions = None

    return cropped_positions


def pad_default_cropping(cropped_image_size, annotation_type):
    """
    Computes the default cropped positions for the FaceCropper used in PAD applications,
    proportionally to the target image size


    Parameters
    ----------
    cropped_image_size : tuple
        A tuple (HEIGHT, WIDTH) describing the target size of the cropped image.

    annotation_type: str
        Type of annotations. Possible values are: `bounding-box`, `eyes-center` and None, or a combination of those as a list

    Returns
    -------

    cropped_positions:
        The dictionary of cropped positions that will be feeded to the FaceCropper, or a list of such dictionaries if
        ``annotation_type`` is a list
    """
    if cropped_image_size[0] != cropped_image_size[1]:
        logger.warning(
            "PAD cropping is designed for a square cropped image size. Got : {}".format(
                cropped_image_size
            )
        )
    else:
        face_size = cropped_image_size[0]

    cropped_positions = {}

    if annotation_type == "bounding-box":
        cropped_positions.update(
            {
                "topleft": (0, 0),
                "bottomright": cropped_image_size,
            }
        )

    if annotation_type in ["bounding-box", "eyes-center"]:
        # We also add cropped eye positions if `bounding-box`, to work with the BoundingBoxCropAnnotator

        eyes_distance = (face_size + 1) / 2.0
        eyes_center = (face_size / 4.0, (face_size - 0.5) / 2.0)
        right_eye = (eyes_center[0], eyes_center[1] - eyes_distance / 2)
        left_eye = (eyes_center[0], eyes_center[1] + eyes_distance / 2)
        cropped_positions.update({"reye": right_eye, "leye": left_eye})

    else:
        logger.warning(
            f"Annotation type {annotation_type} is not supported. Input images will be fully scaled."
        )
        cropped_positions = None

    return cropped_positions


def make_cropper(
    cropped_image_size,
    cropped_positions,
    fixed_positions=None,
    color_channel="rgb",
    annotator=None,
    **kwargs,
):
    """
    Solve the face FaceCropper and additionally returns the necessary
    transform_extra_arguments for wrapping the cropper with a SampleWrapper.

    """
    face_cropper = face_crop_solver(
        cropped_image_size=cropped_image_size,
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        annotator=annotator,
        color_channel=color_channel,
        dtype="float64",
        **kwargs,
    )

    transform_extra_arguments = (
        None
        if (cropped_positions is None or fixed_positions is not None)
        else (("annotations", "annotations"),)
    )

    return face_cropper, transform_extra_arguments


def embedding_transformer(
    cropped_image_size,
    embedding,
    cropped_positions,
    fixed_positions=None,
    color_channel="rgb",
    annotator=None,
    **kwargs,
):
    """
    Creates a pipeline composed by a FaceCropper and an Embedding extractor.
    This transformer is suited for Facenet based architectures

    .. warning::
       This will resize images to the requested `image_size`

    """

    face_cropper, transform_extra_arguments = make_cropper(
        cropped_image_size=cropped_image_size,
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel=color_channel,
        annotator=annotator,
        **kwargs,
    )

    # Support None and "passthrough" Estimators
    if embedding is not None and type(embedding) is not str:
        embedding = wrap(["sample"], embedding)

    transformer = Pipeline(
        [
            (
                "cropper",
                wrap(
                    ["sample"],
                    face_cropper,
                    transform_extra_arguments=transform_extra_arguments,
                ),
            ),
            ("embedding", embedding),
        ]
    )

    return transformer


def face_crop_solver(
    cropped_image_size,
    cropped_positions=None,
    color_channel="rgb",
    fixed_positions=None,
    annotator=None,
    dtype="uint8",
    **kwargs,
):
    """
    Decide which face cropper to use.
    """
    # If there's not cropped positions, just resize
    if cropped_positions is None:
        return Scale(cropped_image_size)
    else:
        # Detects the face and crops it without eye detection

        if isinstance(cropped_positions, list):

            # TODO: This is a hack to support multiple annotations for left, right and eyes center profile
            # We need to do a more elegant solution

            croppers = []
            for positions in cropped_positions:
                if "leye" in positions and "reye" in positions:
                    cropper = FaceCrop(
                        cropped_image_size=cropped_image_size,
                        cropped_positions=positions,
                        fixed_positions=fixed_positions,
                        color_channel=color_channel,
                        annotator=annotator,
                        dtype=dtype,
                        cropper=FaceEyesNorm(
                            positions,
                            cropped_image_size,
                            annotation_type="eyes-center",
                        ),
                    )

                elif "leye" in positions and "mouth" in positions:
                    cropper = FaceCrop(
                        cropped_image_size=cropped_image_size,
                        cropped_positions=positions,
                        fixed_positions=fixed_positions,
                        color_channel=color_channel,
                        annotator=annotator,
                        dtype=dtype,
                        cropper=FaceEyesNorm(
                            positions,
                            cropped_image_size,
                            annotation_type="left-profile",
                        ),
                    )
                elif "reye" in positions and "mouth" in positions:
                    cropper = FaceCrop(
                        cropped_image_size=cropped_image_size,
                        cropped_positions=positions,
                        fixed_positions=fixed_positions,
                        color_channel=color_channel,
                        annotator=annotator,
                        dtype=dtype,
                        cropper=FaceEyesNorm(
                            positions,
                            cropped_image_size,
                            annotation_type="right-profile",
                        ),
                    )

                else:
                    raise ValueError(
                        f"Unsupported list of annotations {cropped_positions}"
                    )

                croppers.append(cropper)

            return MultiFaceCrop(croppers)
        else:
            # If the eyes annotations are provided
            if (
                "topleft" in cropped_positions
                or "bottomright" in cropped_positions
            ):
                eyes_cropper = FaceEyesNorm(
                    cropped_positions, cropped_image_size
                )
                return BoundingBoxAnnotatorCrop(
                    eyes_cropper=eyes_cropper,
                    annotator=annotator,
                )

            else:
                return FaceCrop(
                    cropped_image_size=cropped_image_size,
                    cropped_positions=cropped_positions,
                    color_channel=color_channel,
                    fixed_positions=fixed_positions,
                    dtype=dtype,
                    annotator=annotator,
                    **kwargs,
                )


def get_default_cropped_positions(mode, cropped_image_size, annotation_type):
    """
    Computes the default cropped positions for the FaceCropper,
    proportionally to the target image size


    Parameters
    ----------
    mode: str
        Which default cropping to use. Available modes are : `legacy` (legacy baselines), `facenet`, `arcface`,
        and `pad`.

    cropped_image_size : tuple
        A tuple (HEIGHT, WIDTH) describing the target size of the cropped image.

    annotation_type: str
        Type of annotations. Possible values are: `bounding-box`, `eyes-center` and None, or a combination of those as a list

    Returns
    -------

    cropped_positions:
        The dictionary of cropped positions that will be feeded to the FaceCropper, or a list of such dictionaries if
        ``annotation_type`` is a list
    """
    if mode == "legacy":
        return legacy_default_cropping(cropped_image_size, annotation_type)
    elif mode in ["dnn", "facenet", "arcface"]:
        return dnn_default_cropping(cropped_image_size, annotation_type)
    elif mode == "pad":
        return pad_default_cropping(cropped_image_size, annotation_type)
    else:
        raise ValueError("Unknown default cropping mode `{}`".format(mode))
