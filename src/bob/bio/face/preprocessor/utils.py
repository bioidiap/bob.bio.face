import bob.bio.base


def load_cropper(face_cropper):

    if face_cropper is None:
        cropper = None
    elif isinstance(face_cropper, str):
        cropper = bob.bio.base.load_resource(face_cropper, "preprocessor")
    else:
        cropper = face_cropper

    return cropper
