from bob.bio.face.embeddings.mxnet import arcface_baseline
from bob.bio.face.utils import lookup_config_from_database

annotation_type, fixed_positions, memory_demanding = lookup_config_from_database(
    locals().get("database")
)


def load(annotation_type, fixed_positions=None):

    return arcface_baseline(
        embedding=ArcFaceInsightFace_LResNet100(memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


pipeline = load(annotation_type, fixed_positions)
