import copy
import logging
import os

from functools import partial

import pandas as pd

import bob.io.base

from bob.bio.base.pipelines.abstract_classes import Database
from bob.extension import rc
from bob.pipelines import hash_string
from bob.pipelines.sample import DelayedSample, SampleSet

logger = logging.getLogger(__name__)


def _make_sample_from_template_row(row, image_directory):

    # Appending this key, so we can handle parallel writting done correctly
    # paying the penalty of having duplicate files
    key = os.path.splitext(row["FILENAME"])[0] + "-" + str(row["SUBJECT_ID"])

    return DelayedSample(
        load=partial(
            bob.io.base.load, os.path.join(image_directory, row["FILENAME"])
        ),
        reference_id=str(row["TEMPLATE_ID"]),
        subject_id=str(row["SUBJECT_ID"]),
        key=key,
        # gender=row["GENDER"],
        # indoor_outdoor=row["INDOOR_OUTDOOR"],
        # skintone=row["SKINTONE"],
        # yaw=row["YAW"],
        # rool=row["ROLL"],
        # occ1=row["OCC1"],
        # occ2=row["OCC2"],
        # occ3=row["OCC3"],
        # occ4=row["OCC4"],
        # occ5=row["OCC5"],
        # occ6=row["OCC6"],
        # occ7=row["OCC7"],
        # occ8=row["OCC8"],
        # occ9=row["OCC9"],
        # occ10=row["OCC10"],
        # occ11=row["OCC11"],
        # occ12=row["OCC12"],
        # occ13=row["OCC13"],
        # occ14=row["OCC14"],
        # occ15=row["OCC15"],
        # occ16=row["OCC16"],
        # occ17=row["OCC17"],
        # occ18=row["OCC18"],
        annotations={
            "topleft": (float(row["FACE_Y"]), float(row["FACE_X"])),
            "bottomright": (
                float(row["FACE_Y"]) + float(row["FACE_HEIGHT"]),
                float(row["FACE_X"]) + float(row["FACE_WIDTH"]),
            ),
            "size": (float(row["FACE_HEIGHT"]), float(row["FACE_WIDTH"])),
        },
    )


def _make_sample_set_from_template_group(template_group, image_directory):

    samples = list(
        template_group.apply(
            _make_sample_from_template_row,
            axis=1,
            image_directory=image_directory,
        )
    )
    return SampleSet(
        samples,
        reference_id=samples[0].reference_id,
        subject_id=samples[0].subject_id,
        key=samples[0].reference_id,
    )


class IJBCDatabase(Database):
    """

    This package contains the access API and descriptions for the IARPA Janus Benchmark C -- IJB-C database.
    The actual raw data can be downloaded from the original web page: http://www.nist.gov/programs-projects/face-challenges (note that not everyone might be eligible for downloading the data).

    Included in the database, there are list files defining verification as well as closed- and open-set identification protocols.
    For verification, two different protocols are provided.
    For the ``1:1`` protocol, gallery and probe templates are combined using several images and video frames for each subject.
    Compared gallery and probe templates share the same gender and skin tone -- these have been matched to make the comparisons more realistic and difficult.

    For closed-set identification, the gallery of the ``1:1`` protocol is used, while probes stem from either only images, mixed images and video frames, or plain videos.
    For open-set identification, the same probes are evaluated, but the gallery is split into two parts, either of which is left out to provide unknown probe templates, i.e., probe templates with no matching subject in the gallery.
    In any case, scores are computed between all (active) gallery templates and all probes.

    The IJB-C dataset provides additional evaluation protocols for face detection and clustering, but these are (not yet) part of this interface.


    .. warning::

      To use this dataset protocol, you need to have the original files of the IJBC datasets.
      Once you have it downloaded, please run the following command to set the path for Bob

        .. code-block:: sh

            bob config set bob.bio.face.ijbc.directory [IJBC PATH]


    The code below allows you to fetch the galery and probes of the "1:1" protocol.

    .. code-block:: python

        >>> from bob.bio.face.database import IJBCDatabase
        >>> ijbc = IJBCDatabase(protocol="test1")
        >>>
        >>> # Fetching the gallery
        >>> references = ijbc.references()
        >>> # Fetching the probes
        >>> probes = ijbc.probes()

    """

    def __init__(
        self,
        protocol,
        original_directory=rc.get("bob.bio.face.ijbc.directory"),
        **kwargs,
    ):

        if original_directory is None or not os.path.exists(original_directory):
            raise ValueError(
                f"Invalid or non existent `original_directory`: {original_directory}"
            )

        self._check_protocol(protocol)
        super().__init__(
            name="ijbc",
            protocol=protocol,
            score_all_vs_all=False,
            annotation_type="bounding-box",
            fixed_positions=None,
            memory_demanding=True,
        )

        self.image_directory = os.path.join(original_directory, "images")
        self.protocol_directory = os.path.join(original_directory, "protocols")
        self._cached_probes = None
        self._cached_references = None
        self.hash_fn = hash_string

        self._load_metadata(protocol)

        # For the test4 protocols
        if "test4" in protocol:
            self.score_all_vs_all = True

    def _load_metadata(self, protocol):
        # Load CSV files
        if protocol == "test1" or protocol == "test2":
            self.reference_templates = pd.read_csv(
                os.path.join(
                    self.protocol_directory, protocol, "enroll_templates.csv"
                )
            )

            self.probe_templates = pd.read_csv(
                os.path.join(
                    self.protocol_directory, protocol, "verif_templates.csv"
                )
            )

            self.matches = pd.read_csv(
                os.path.join(self.protocol_directory, protocol, "match.csv"),
                names=["ENROLL_TEMPLATE_ID", "VERIF_TEMPLATE_ID"],
            ).astype("str")

            # TODO: temporarily disabling the metadata
            """
            self.metadata = pd.read_csv(
                os.path.join(self.protocol_directory, "ijbc_metadata_with_age.csv"),
                usecols=[
                    "SUBJECT_ID",
                    "FILENAME",
                    "FACE_X",
                    "FACE_Y",
                    "FACE_WIDTH",
                    "FACE_HEIGHT",
                    "SIGHTING_ID",
                    "FACIAL_HAIR",
                    "AGE",
                    "INDOOR_OUTDOOR",
                    "SKINTONE",
                    "GENDER",
                    "YAW",
                    "ROLL",
                ]
                + [f"OCC{i}" for i in range(1, 19)],
            )

            # LEFT JOIN WITH METADATA
            self.probe_templates = pd.merge(
                self.probe_templates,
                self.metadata,
                on=[
                    "SUBJECT_ID",
                    "FILENAME",
                    "FACE_X",
                    "FACE_Y",
                    "FACE_WIDTH",
                    "FACE_HEIGHT",
                ],
                how="left",
            )

            # LEFT JOIN WITH METADATA
            self.reference_templates = pd.merge(
                self.reference_templates,
                self.metadata,
                on=[
                    "SUBJECT_ID",
                    "FILENAME",
                    "FACE_X",
                    "FACE_Y",
                    "FACE_WIDTH",
                    "FACE_HEIGHT",
                ],
                how="left",
            )
            """

        elif "test4" in protocol:
            gallery_file = (
                "gallery_G1.csv" if "G1" in protocol else "gallery_G2.csv"
            )

            self.reference_templates = pd.read_csv(
                os.path.join(self.protocol_directory, "test4", gallery_file)
            )

            self.probe_templates = pd.read_csv(
                os.path.join(self.protocol_directory, "test4", "probes.csv")
            )

            self.matches = None

        else:
            raise ValueError(
                f"Protocol `{protocol}` not supported. We do accept merge requests :-)"
            )

    def background_model_samples(self):
        return None

    def probes(self, group="dev"):
        self._check_group(group)
        if self._cached_probes is None:

            logger.info(
                "Loading probes. This operation might take some minutes"
            )

            self._cached_probes = list(
                self.probe_templates.groupby("TEMPLATE_ID").apply(
                    _make_sample_set_from_template_group,
                    image_directory=self.image_directory,
                )
            )

            # Wiring probes with references
            if self.protocol == "test1" or self.protocol == "test2":
                # Link probes to the references they have to be compared with
                # We might make that faster if we manage to write it as a Panda instruction
                grouped_matches = self.matches.groupby("VERIF_TEMPLATE_ID")
                for probe_sampleset in self._cached_probes:
                    probe_sampleset.references = list(
                        grouped_matches.get_group(probe_sampleset.reference_id)[
                            "ENROLL_TEMPLATE_ID"
                        ]
                    )
            elif "test4" in self.protocol:
                references = [s.reference_id for s in self.references()]
                # You compare with all biometric references
                for probe_sampleset in self._cached_probes:
                    probe_sampleset.references = copy.deepcopy(references)
                    pass

            else:
                raise ValueError(f"Invalid protocol: {self.protocol}")

        return self._cached_probes

    def references(self, group="dev"):
        self._check_group(group)
        if self._cached_references is None:

            logger.info(
                "Loading templates. This operation might take some minutes"
            )

            self._cached_references = list(
                self.reference_templates.groupby("TEMPLATE_ID").apply(
                    _make_sample_set_from_template_group,
                    image_directory=self.image_directory,
                )
            )

        return self._cached_references

    def all_samples(self, group="dev"):
        self._check_group(group)

        return self.references() + self.probes()

    def groups(self):
        return ["dev"]

    def protocols(self):
        return ["test1", "test2", "test4-G1", "test4-G2"]

    def _check_protocol(self, protocol):
        assert (
            protocol in self.protocols()
        ), "Invalid protocol `{}` not in {}".format(protocol, self.protocols())

    def _check_group(self, group):
        assert group in self.groups(), "Invalid group `{}` not in {}".format(
            group, self.groups()
        )
