from sklearn.pipeline import make_pipeline

from bob.bio.base.pipelines import Distance, PipelineSimple
from bob.bio.face.utils import lookup_config_from_database
from bob.pipelines import wrap
from bob.pipelines.transformers import SampleLinearize

(
    annotation_type,
    fixed_positions,
    memory_demanding,
) = lookup_config_from_database()

from sklearn.base import BaseEstimator, TransformerMixin

from bob.bio.face.color import rgb_to_gray


class ToGray(TransformerMixin, BaseEstimator):
    def transform(self, X, annotations=None):
        return [rgb_to_gray(data)[0:10, 0:10] for data in X]

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

    return PipelineSimple(transformer, algorithm)


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
