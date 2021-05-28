from bob.bio.face.embeddings.mxnet import ArcFaceInsightFace_LResNet100
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

annotation_type, fixed_positions, memory_demanding = lookup_config_from_database(
    locals().get("database")
)


def load(annotation_type, fixed_positions=None):
    transformer = embedding_transformer_112x112(
        ArcFaceInsightFace_LResNet100(memory_demanding=memory_demanding),
        annotation_type,
        fixed_positions,
        color_channel="rgb",
    )

    return arcface_baseline(
        embedding=ArcFaceInsightFace(memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
