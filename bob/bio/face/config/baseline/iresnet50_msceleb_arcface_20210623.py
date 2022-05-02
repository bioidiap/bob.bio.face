from bob.bio.face.embeddings.tensorflow import (
    iresnet50_msceleb_arcface_20210623,
)
from bob.bio.face.utils import lookup_config_from_database

(
    annotation_type,
    fixed_positions,
    memory_demanding,
) = lookup_config_from_database(locals().get("database"))


def load(annotation_type, fixed_positions=None, memory_demanding=None):
    return iresnet50_msceleb_arcface_20210623(
        annotation_type, fixed_positions, memory_demanding
    )


pipeline = load(annotation_type, fixed_positions, memory_demanding)
transformer = pipeline.transformer
