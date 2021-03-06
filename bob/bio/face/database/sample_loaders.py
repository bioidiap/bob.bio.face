#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""  Sample and Metatada loaders"""


from bob.pipelines import DelayedSample, Sample, SampleSet
from sklearn.base import TransformerMixin, BaseEstimator


def find_attribute(x, attribute):
    if hasattr(x, attribute):
        return getattr(x, attribute)
    else:
        ValueError(f"Attribute not found in the dataset: {attribute}")


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

        annotated_samples = []
        for x in X:
            eyes = {
                "leye": (
                    float(find_attribute(x, "leye_x")),
                    float(find_attribute(x, "leye_y")),
                ),
                "reye": (
                    float(find_attribute(x, "reye_x")),
                    float(find_attribute(x, "reye_y")),
                ),
            }

            sample = DelayedSample(x._load, parent=x, annotations=eyes)
            [delattr(sample, a) for a in ["leye_x", "leye_y", "reye_x", "reye_y"]]
            annotated_samples.append(sample)

        return annotated_samples


class MultiposeAnnotations(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {
            "stateless": True,
            "requires_fit": False,
        }

    def transform(self, X):

        annotated_samples = []
        for x in X:
            annotations = dict()
            if find_attribute(x, "leye_x") != "" and find_attribute(x, "reye_x") != "":
                # Normal profile
                annotations = {
                    "leye": (
                        float(find_attribute(x, "leye_x")),
                        float(find_attribute(x, "leye_y")),
                    ),
                    "reye": (
                        float(find_attribute(x, "reye_x")),
                        float(find_attribute(x, "reye_y")),
                    ),
                }
            elif (
                find_attribute(x, "leye_x") != "" and find_attribute(x, "reye_x") == ""
            ):
                # Left profile
                annotations = {
                    "leye": (
                        float(find_attribute(x, "leye_x")),
                        float(find_attribute(x, "leye_y")),
                    ),
                    "mouth": (
                        float(find_attribute(x, "mouthl_x")),
                        float(find_attribute(x, "mouthl_y")),
                    ),
                }
            elif (
                find_attribute(x, "leye_x") == "" and find_attribute(x, "reye_x") != ""
            ):
                # Right profile
                annotations = {
                    "reye": (
                        float(find_attribute(x, "reye_x")),
                        float(find_attribute(x, "reye_y")),
                    ),
                    "mouth": (
                        float(find_attribute(x, "mouthr_x")),
                        float(find_attribute(x, "mouthr_y")),
                    ),
                }
            else:
                raise ValueError("Annotations not available")

            sample = DelayedSample(x._load, parent=x, annotations=annotations)
            [
                delattr(sample, a)
                for a in [
                    "reye_x",
                    "reye_y",
                    "leye_x",
                    "leye_y",
                    "nose_x",
                    "nose_y",
                    "mouthr_x",
                    "mouthr_y",
                    "mouthl_x",
                    "mouthl_y",
                    "chin_x",
                    "chin_y",
                ]
            ]

            annotated_samples.append(sample)

        return annotated_samples
