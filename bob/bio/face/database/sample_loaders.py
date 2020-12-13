#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""  Sample and Metatada loaders"""


from bob.pipelines import DelayedSample, Sample, SampleSet
from sklearn.base import TransformerMixin, BaseEstimator


class EyesAnnotations(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {
            "stateless": True,
            "requires_fit": False,
        }

    def transform(self, X):
        """
        Convert  leye_x, leye_y, reye_x, reye_y attributes to `annotations = (leye, reye)`
        """

        def find_attribute(x, attribute):
            if hasattr(x, attribute):
                return getattr(x, attribute)
            else:
                ValueError(f"Attribute not found in the dataset: {attribute}")

        annotated_samples = []
        for x in X:
            eyes = {
                "leye": (find_attribute, (x, "leye_x"), find_attribute(x, "leye_y")),
                "reye": (find_attribute(x, "reye_x"), find_attribute(x, "reye_y")),
            }

            sample = DelayedSample(x._load, parent=x, annotations=eyes)
            [delattr(sample, a) for a in ["leye_x", "leye_y", "reye_x", "reye_y"]]
            annotated_samples.append(sample)

        return annotated_samples
