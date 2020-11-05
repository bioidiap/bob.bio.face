from bob.bio.face.embeddings import ArcFace_InsightFaceTF
from bob.bio.face.config.baseline.helpers import embedding_transformer_112x112
from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
)


if "database" in locals():
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
else:
    annotation_type = None
    fixed_positions = None


def load(annotation_type, fixed_positions=None):
    transformer = embedding_transformer_112x112(
        ArcFace_InsightFaceTF(), annotation_type, fixed_positions
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)

pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer