from bob.bio.face.embeddings.opencv import vgg16_oxford_baseline
from bob.bio.face.utils import lookup_config_from_database

annotation_type, fixed_positions, _ = lookup_config_from_database(
    locals().get("database")
)


def load(annotation_type, fixed_positions=None):
    return vgg16_oxford_baseline(annotation_type, fixed_positions)


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
