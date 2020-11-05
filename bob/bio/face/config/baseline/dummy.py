from bob.pipelines import wrap
from sklearn.pipeline import make_pipeline
from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
)
from bob.pipelines.transformers import SampleLinearize

if "database" in locals():
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
else:
    annotation_type = None
    fixed_positions = None


import bob.ip.color
from sklearn.base import TransformerMixin, BaseEstimator
class ToGray(TransformerMixin, BaseEstimator):

    def transform(self, X, annotations=None):
        return [bob.ip.color.rgb_to_gray(data)[0:10,0:10] for data in X]

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def fit(self, X, y=None):
        return self


def load(annotation_type, fixed_positions=None):

    transform_extra_arguments = (("annotations", "annotations"),)

    transformer = make_pipeline(
        wrap(
            ["sample"],
            ToGray(),
            transform_extra_arguments=transform_extra_arguments,
        ),
        SampleLinearize(),
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
