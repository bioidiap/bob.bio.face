from bob.bio.face.embeddings.tf2_inception_resnet import (
    FaceNetSanderberg_20170512_110547,
)
from bob.bio.face.config.baseline.helpers import embedding_transformer_160x160
from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
)

memory_demanding = False
if "database" in locals():
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
    memory_demanding = (
        database.memory_demanding if hasattr(database, "memory_demanding") else False
    )

else:
    annotation_type = None
    fixed_positions = None


def load(annotation_type, fixed_positions=None):

    transformer = embedding_transformer_160x160(
        FaceNetSanderberg_20170512_110547(memory_demanding=memory_demanding),
        annotation_type,
        fixed_positions,
    )
    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
