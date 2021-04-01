from bob.bio.face.embeddings.mxnet_models import ArcFaceInsightFace
from bob.bio.face.config.baseline.helpers import embedding_transformer_112x112
from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
)


if "database" in locals():
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
    memory_demanding = (
        database.memory_demanding if hasattr(database, "memory_demanding") else False
    )
else:
    annotation_type = None
    fixed_positions = None
    memory_demanding = False


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
