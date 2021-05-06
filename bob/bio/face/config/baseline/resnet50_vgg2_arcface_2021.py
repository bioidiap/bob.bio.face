from bob.bio.face.embeddings.resnet50 import Resnet50_VGG2_ArcFace_2021
from bob.bio.face.config.baseline.helpers import (
    embedding_transformer_112x112,
    lookup_config_from_database,
)
from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
)

annotation_type, fixed_positions, memory_demanding = lookup_config_from_database()


def load(annotation_type, fixed_positions=None):
    transformer = embedding_transformer_112x112(
        Resnet50_VGG2_ArcFace_2021(memory_demanding=memory_demanding),
        annotation_type,
        fixed_positions,
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
