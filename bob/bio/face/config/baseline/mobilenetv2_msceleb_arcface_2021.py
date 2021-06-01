from bob.bio.face.embeddings.mobilenet_v2 import MobileNetv2_MsCeleb_ArcFace_2021
from bob.bio.face.utils import lookup_config_from_database
from bob.bio.face.config.baseline.templates import arcface_baseline


annotation_type, fixed_positions, memory_demanding = lookup_config_from_database(
    locals().get("database")
)


def load(annotation_type, fixed_positions=None):

    return arcface_baseline(
        embedding=MobileNetv2_MsCeleb_ArcFace_2021(memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
