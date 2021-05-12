from bob.extension import rc
from bob.bio.face.embeddings.tf2_inception_resnet import InceptionResnetv2
from bob.bio.face.preprocessor import FaceCrop
from bob.bio.face.config.baseline.helpers import (
    lookup_config_from_database,
    dnn_default_cropping,
    embedding_transformer,
)

from sklearn.pipeline import make_pipeline
from bob.pipelines.wrappers import wrap
from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
)

annotation_type, fixed_positions, memory_demanding = lookup_config_from_database()


def load(annotation_type, fixed_positions=None):
    # DEFINE CROPPING
    cropped_image_size = (160, 160)
    cropped_positions = dnn_default_cropping(cropped_image_size, annotation_type)

    extractor_path = rc["bob.bio.face.tf2.casia-webface-inception-v2"]
    embedding = InceptionResnetv2(
        checkpoint_path=extractor_path, memory_demanding=memory_demanding
    )
    # ASSEMBLE TRANSFORMER
    transformer = embedding_transformer(
        cropped_image_size=cropped_image_size,
        embedding=embedding,
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="rgb",
        annotator="mtcnn",
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
