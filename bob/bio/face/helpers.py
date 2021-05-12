from bob.bio.face.preprocessor import FaceCrop, MultiFaceCrop, Scale


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
                annotation=annotator,
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
