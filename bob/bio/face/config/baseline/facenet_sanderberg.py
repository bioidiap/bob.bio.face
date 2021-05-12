from bob.bio.face.embeddings.tf2_inception_resnet import (
    FaceNetSanderberg_20170512_110547,
)
from bob.bio.face.config.baseline.helpers import lookup_config_from_database
from bob.bio.face.config.baseline.templates import facenet_baseline

annotation_type, fixed_positions, memory_demanding = lookup_config_from_database()


def load(annotation_type, fixed_positions=None):
    return facenet_baseline(
        embedding=FaceNetSanderberg_20170512_110547(memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
