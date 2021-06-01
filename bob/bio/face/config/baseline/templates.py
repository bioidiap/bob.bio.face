from bob.bio.face.utils import (
    dnn_default_cropping,
    embedding_transformer,
)
from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
)


def arcface_baseline(embedding, annotation_type, fixed_positions=None):
    # DEFINE CROPPING
    cropped_image_size = (112, 112)
    if annotation_type == "eyes-center":
        # Hard coding eye positions for backward consistency
        cropped_positions = {
            "leye": (55, 81),
            "reye": (55, 42),
        }
    else:
        cropped_positions = dnn_default_cropping(cropped_image_size, annotation_type)

    transformer = embedding_transformer(
        cropped_image_size=cropped_image_size,
        embedding=embedding,
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="rgb",
        annotator="mtcnn",
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)


def facenet_baseline(embedding, annotation_type, fixed_positions=None):
    # DEFINE CROPPING
    cropped_image_size = (160, 160)
    cropped_positions = dnn_default_cropping(cropped_image_size, annotation_type)

    # ASSEMBLE TRANSFORMER
    transformer = embedding_transformer(
        cropped_image_size=cropped_image_size,
        embedding=embedding,
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="rgb",
        annotator="mtcnn",
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)
