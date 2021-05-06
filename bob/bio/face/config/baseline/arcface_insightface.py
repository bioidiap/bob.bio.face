from bob.bio.face.embeddings.mxnet_models import ArcFaceInsightFace
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
        ArcFaceInsightFace(memory_demanding=memory_demanding),
        annotation_type,
        fixed_positions,
        color_channel="rgb",
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
