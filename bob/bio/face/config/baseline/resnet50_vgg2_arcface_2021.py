from bob.bio.face.embeddings.resnet50 import Resnet50_VGG2_ArcFace_2021
from bob.bio.face.config.baseline.helpers import (
    lookup_config_from_database,
    dnn_default_cropping,
    embedding_transformer,
)
from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
)

annotation_type, fixed_positions, memory_demanding = lookup_config_from_database()


def load(annotation_type, fixed_positions=None):
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
        embedding=Resnet50_VGG2_ArcFace_2021(memory_demanding=memory_demanding),
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="rgb",
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
