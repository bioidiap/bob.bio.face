from bob.extension import rc
from bob.bio.face.embeddings.tf2_inception_resnet import InceptionResnetv2
from bob.bio.face.config.baseline.helpers import lookup_config_from_database
from bob.bio.face.config.baseline.templates import facenet_baseline

annotation_type, fixed_positions, memory_demanding = lookup_config_from_database()


def load(annotation_type, fixed_positions=None):
    extractor_path = rc["bob.bio.face.tf2.casia-webface-inception-v2"]
    embedding = InceptionResnetv2(
        checkpoint_path=extractor_path, memory_demanding=memory_demanding
    )
    return facenet_baseline(
        embedding=embedding,
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
