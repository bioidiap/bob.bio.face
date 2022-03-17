from bob.bio.face.embeddings.pytorch import MobileFaceNet
from bob.bio.face.utils import lookup_config_from_database


annotation_type, fixed_positions, memory_demanding = lookup_config_from_database(
    locals().get("database")
)


def load(annotation_type, fixed_positions=None, memory_demanding=False):
    return MobileFaceNet(annotation_type, fixed_positions, memory_demanding)


pipeline = load(annotation_type, fixed_positions, memory_demanding)

transformer = pipeline.transformer
transformer = pipeline.transformer
