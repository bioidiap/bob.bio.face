from bob.bio.face.embeddings.pytorch import AttentionNet
from bob.bio.face.utils import lookup_config_from_database

(
    annotation_type,
    fixed_positions,
    memory_demanding,
) = lookup_config_from_database(locals().get("database"))


def load(annotation_type, fixed_positions=None, memory_demanding=False):
    return AttentionNet(annotation_type, fixed_positions, memory_demanding)


pipeline = load(annotation_type, fixed_positions, memory_demanding)

transformer = pipeline.transformer
