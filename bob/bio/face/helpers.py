from bob.bio.face.preprocessor import FaceCrop, MultiFaceCrop, Scale
import bob.bio.face.config.baseline.helpers as helpers

def face_crop_solver(
    cropped_image_size,
    cropped_positions=None,
    color_channel="rgb",
    fixed_positions=None,
    annotator=None,
    dtype="uint8",
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
            return MultiFaceCrop(
                cropped_image_size=cropped_image_size,
                cropped_positions_list=cropped_positions,
                fixed_positions_list=fixed_positions,
                color_channel=color_channel,
                dtype=dtype,
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
        return helpers.legacy_default_cropping(cropped_image_size, annotation_type)
    elif mode in ["dnn", "facenet", "arcface"]:
        return helpers.dnn_default_cropping(cropped_image_size, annotation_type)
    elif mode == "pad":
        return helpers.pad_default_cropping(cropped_image_size, annotation_type)
    else:
        raise ValueError("Unknown default cropping mode `{}`".format(mode))
