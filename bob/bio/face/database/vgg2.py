#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import logging

logger = logging.getLogger(__name__)


"""
  VGG2 database implementation
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDatasetZTNorm, CSVToSampleLoaderBiometrics
from bob.extension import rc
from bob.extension.download import get_file
from bob.pipelines import DelayedSample


class VGG2Annotations(TransformerMixin, BaseEstimator):
    """
    Decode VGG2 Annotations.

    VGG2 has 5 landmarks annnotated and the are the following: two for eyes, one for the nose and two for the mouth

    """

    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {
            "requires_fit": False,
        }

    def transform(self, X):
        """
        Convert  leye_x, leye_y, reye_x, reye_y attributes to `annotations = (leye, reye)`
        """

        annotated_samples = []

        for x in X:
            annotations = {
                "leye": (
                    float(x.leye_y),
                    float(x.leye_x),
                ),
                "reye": (
                    float(x.reye_y),
                    float(x.reye_x),
                ),
                "nose": (
                    float(x.nose_y),
                    float(x.nose_x),
                ),
                "lmouth": (
                    float(x.lmouth_y),
                    float(x.lmouth_x),
                ),
                "rmouth": (
                    float(x.rmouth_y),
                    float(x.rmouth_x),
                ),
                "topleft": (
                    float(x.face_y),
                    float(x.face_x),
                ),
                "size": (
                    float(x.face_h),
                    float(x.face_w),
                ),
            }

            sample = DelayedSample.from_sample(x, annotations=annotations)
            # Cleaning up
            [
                delattr(sample, a)
                for a in [
                    "leye_x",
                    "leye_y",
                    "reye_x",
                    "reye_y",
                    "nose_y",
                    "nose_x",
                    "face_y",
                    "face_x",
                    "face_h",
                    "face_w",
                    "lmouth_y",
                    "lmouth_x",
                    "rmouth_y",
                    "rmouth_x",
                ]
            ]
            annotated_samples.append(sample)

        return annotated_samples


class VGG2Database(CSVDatasetZTNorm):
    """
    The VGG2 Dataset is composed of 9131 people split into two sets.
    The training set contains 8631 identities, while the test set contains 500 identities.

    As metadata, this dataset contains the gender labels "m" and "f" for, respectively, male and female.
    It also contains the following race labels:

        - A: Asian in general (Chinese, Japanese, Filipino, Korean, Polynesian, Indonesian, Samoan, or any other Pacific Islander
        - B: A person having origins in any of the black racial groups of Africa
        - I: American Indian, Asian Indian, Eskimo, or Alaskan native
        - U: Of indeterminable race
        - W: Caucasian, Mexican, Puerto Rican, Cuban, Central or South American, or other Spanish culture or origin, Regardless of race
        - N: None of the above

    Race labels are taken from: MasterEBTSv10.0.809302017_Final.pdf.

    This dataset also contains sets for `T-Norm` and `Z-Norm`, normalization.


    We provide four protocols; `vgg2-short`, `vgg2-full`,`vgg2-short-with-eval`, `vgg2-full-with-eval`.
    The `vgg2-short` and `vgg2-full` present the sample amount of identities but
    varies with respect to the number of samples per identity.
    The `vgg2-full` preserves the number of samples per identity from the original dataset.
    On the other hand, the `vgg2-short` presents 10 samples per identity at the probe and training sets.
    With that the training set of `vgg2-short` contains 86'310 samples instead of 3'141'890 samples
    from `vgg2-full`.
    The protocols with the suffix `-with-eval`, splits the orinal test set into a dev and eval sets
    containing 250 identities each.


    All the landmarks and face crops provided in the original dataset is provided with this inteface.

    .. warning::

        To use this dataset protocol, you need to have the original files of the VGG2 dataset.
        Once you have it downloaded, please run the following command to set the path for Bob

        .. code-block:: sh

            bob config set bob.bio.face.vgg2.directory [VGG2 PATH]
            bob config set bob.bio.face.vgg2.extension [VGG2 EXTENSION]

    For more information check:

    .. code-block:: latex

        @inproceedings{cao2018vggface2,
            title={Vggface2: A dataset for recognising faces across pose and age},
            author={Cao, Qiong and Shen, Li and Xie, Weidi and Parkhi, Omkar M and Zisserman, Andrew},
            booktitle={2018 13th IEEE international conference on automatic face \\& gesture recognition (FG 2018)},
            pages={67--74},
            year={2018},
            organization={IEEE}
        }
    """

    def __init__(
        self,
        protocol,
        dataset_original_directory=rc.get("bob.bio.face.vgg2.directory", ""),
        dataset_original_extension=rc.get(
            "bob.bio.face.vgg2.extension", ".jpg"
        ),
        annotation_type="eyes-center",
        fixed_positions=None,
    ):

        # Downloading model if not exists
        urls = VGG2Database.urls()
        filename = get_file(
            "vgg2.tar.gz", urls, file_hash="4a05d797a326374a6b52bcd8d5a89d48"
        )

        super().__init__(
            name="vgg2",
            dataset_protocol_path=filename,
            protocol=protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=dataset_original_directory,
                    extension=dataset_original_extension,
                ),
                VGG2Annotations(),
            ),
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
        )

    def background_model_samples(self):
        if self.protocol == "vgg2-full":
            logger.warning(
                "This set is very long (3M samples). It might take ~4 minutes to load everything"
            )
        return super().background_model_samples()

    @staticmethod
    def protocols():
        # TODO: Until we have (if we have) a function that dumps the protocols, let's use this one.
        return [
            "vgg2-short",
            "vgg2-full",
            "vgg2-short-with-eval",
            "vgg2-full-with-eval",
        ]

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/vgg2.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/vgg2.tar.gz",
        ]
