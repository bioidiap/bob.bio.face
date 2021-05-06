from bob.extension import rc
from bob.bio.face.embeddings.tf2_inception_resnet import InceptionResnetv2
from bob.bio.face.preprocessor import FaceCrop
from bob.bio.face.config.baseline.helpers import (
    embedding_transformer_default_cropping,
    embedding_transformer,
    lookup_config_from_database,
)

from sklearn.pipeline import make_pipeline
from bob.pipelines.wrappers import wrap
from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
)

annotation_type, fixed_positions, memory_demanding = lookup_config_from_database()


def load(annotation_type, fixed_positions=None):
    CROPPED_IMAGE_SIZE = (160, 160)
    CROPPED_POSITIONS = embedding_transformer_default_cropping(
        CROPPED_IMAGE_SIZE, annotation_type=annotation_type
    )

    extractor_path = rc["bob.bio.face.tf2.casia-webface-inception-v2"]

    embedding = InceptionResnetv2(
        checkpoint_path=extractor_path, memory_demanding=memory_demanding
    )

    transformer = embedding_transformer(
        CROPPED_IMAGE_SIZE,
        embedding,
        annotation_type,
        CROPPED_POSITIONS,
        fixed_positions,
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
