from bob.bio.face.embeddings.pytorch import iresnet34
from bob.bio.face.utils import lookup_config_from_database


annotation_type, fixed_positions, _ = lookup_config_from_database(
    locals().get("database")
)


def load(annotation_type, fixed_positions=None):
    return iresnet34(annotation_type, fixed_positions)


pipeline = load(annotation_type, fixed_positions)

