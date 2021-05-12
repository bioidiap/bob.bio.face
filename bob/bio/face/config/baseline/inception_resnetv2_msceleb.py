from bob.bio.face.embeddings.tf2_inception_resnet import (
    InceptionResnetv2_MsCeleb_CenterLoss_2018,
)
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
    cropped_image_size = (160, 160)
    cropped_positions = dnn_default_cropping(cropped_image_size, annotation_type)

    # ASSEMBLE TRANSFORMER
    transformer = embedding_transformer(
        cropped_image_size=cropped_image_size,
        embedding=InceptionResnetv2_MsCeleb_CenterLoss_2018(
            memory_demanding=memory_demanding
        ),
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="rgb",
        annotator="mtcnn",
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
