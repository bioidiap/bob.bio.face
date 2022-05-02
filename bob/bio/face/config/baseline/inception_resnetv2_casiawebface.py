from bob.bio.face.embeddings.tensorflow import (
    inception_resnet_v2_casia_centerloss_2018,
)
from bob.bio.face.utils import lookup_config_from_database

(
    annotation_type,
    fixed_positions,
    memory_demanding,
) = lookup_config_from_database(locals().get("database"))


def load(annotation_type, fixed_positions=None, memory_demanding=None):
    return inception_resnet_v2_casia_centerloss_2018(
        annotation_type, fixed_positions, memory_demanding
    )


pipeline = load(annotation_type, fixed_positions, memory_demanding)
transformer = pipeline.transformer
