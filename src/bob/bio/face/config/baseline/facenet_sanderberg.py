from bob.bio.face.embeddings.tensorflow import (
    facenet_sanderberg_20170512_110547,
)
from bob.bio.face.utils import lookup_config_from_database

(
    annotation_type,
    fixed_positions,
    memory_demanding,
) = lookup_config_from_database(locals().get("database"))


def load(annotation_type, fixed_positions=None, memory_demanding=False):
    return facenet_sanderberg_20170512_110547(
        annotation_type, fixed_positions, memory_demanding
    )


pipeline = load(annotation_type, fixed_positions, memory_demanding)
transformer = pipeline.transformer
