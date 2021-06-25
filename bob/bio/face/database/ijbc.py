from bob.bio.base.pipelines.vanilla_biometrics.abstract_classes import Database
import pandas as pd
from bob.pipelines.sample import DelayedSample, SampleSet
from bob.extension import rc
import os
import bob.io.image
from functools import partial
import uuid
from bob.pipelines.utils import hash_string


def _make_sample_from_template_row(row, image_directory):

    # Appending this hash, so we can handle parallel writting done correctly
    # paying the penalty of having duplicate files
    hashstr = str(uuid.uuid4())

    return DelayedSample(
        load=partial(bob.io.image.load, os.path.join(image_directory, row["FILENAME"])),
        reference_id=row["TEMPLATE_ID"],
        subject_id=row["SUBJECT_ID"],
        key=os.path.splitext(row["FILENAME"])[0] + "-" + hashstr,
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
            _make_sample_from_template_row, axis=1, image_directory=image_directory
        )
    )
    return SampleSet(
        samples, reference_id=samples[0].reference_id, subject_id=samples[0].subject_id
    )


class IJBCDatabase(Database):
    """

    This package contains the access API and descriptions for the IARPA Janus Benchmark C -- IJB-C database.
    The actual raw data can be downloaded from the original web page: http://www.nist.gov/programs-projects/face-challenges (note that not everyone might be eligible for downloading the data).

    Included in the database, there are list files defining verification as well as closed- and open-set identification protocols.
    For verification, two different protocols are provided.
    For the ``1:1`` protocol, gallery and probe templates are combined using several images and video frames for each subject.
    Compared gallery and probe templates share the same gender and skin tone -- these have been matched to make the comparisions more realistic and difficult.

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
        >>> ijbc = IJBCDatabase()
        >>>
        >>> # Fetching the gallery 
        >>> references = ijbc.references()
        >>> # Fetching the probes 
        >>> probes = ijbc.probes()
    
    """

    def __init__(
        self,
        protocol="1:1",
        original_directory=rc.get("bob.bio.face.ijbc.directory"),
        **kwargs,
    ):

        if original_directory is None or not os.path.exists(original_directory):
            raise ValueError(
                "Invalid or non existant `original_directory`: f{original_directory}"
            )

        self._check_protocol(protocol)
        super().__init__(
            name="ijbc",
            protocol=protocol,
            allow_scoring_with_all_biometric_references=False,
            annotation_type="eyes-center",
            fixed_positions=None,
            memory_demanding=True,
        )

        self.image_directory = os.path.join(original_directory, "images")
        self.protocol_directory = os.path.join(original_directory, "protocols")
        self._cached_probes = None
        self._cached_references = None
        self.hash_fn = hash_string

        self._load_metadata(protocol)

    def _load_metadata(self, protocol):
        # Load CSV files
        if protocol == "1:1":
            self.reference_templates = pd.concat(
                [
                    pd.read_csv(
                        os.path.join(self.protocol_directory, "ijbc_1N_gallery_G1.csv")
                    ),
                    pd.read_csv(
                        os.path.join(self.protocol_directory, "ijbc_1N_gallery_G2.csv")
                    ),
                ]
            )

            self.probe_templates = pd.read_csv(
                os.path.join(self.protocol_directory, "ijbc_1N_probe_mixed.csv")
            )

            self.matches = pd.read_csv(
                os.path.join(self.protocol_directory, "ijbc_11_G1_G2_matches.csv"),
                names=["REFERENCE_TEMPLATE_ID", "PROBE_TEMPLATE_ID"],
            )
        else:
            raise ValueError(
                f"Protocol `{protocol}` not supported. We do accept merge requests :-)"
            )

    def background_model_samples(self):
        return None

    def probes(self, group="dev"):
        self._check_group(group)
        if self._cached_probes is None:
            self._cached_probes = list(
                self.probe_templates.groupby("TEMPLATE_ID").apply(
                    _make_sample_set_from_template_group,
                    image_directory=self.image_directory,
                )
            )

        # Link probes to the references they have to be compared with
        # We might make that faster if we manage to write it as a Panda instruction
        grouped_matches = self.matches.groupby("PROBE_TEMPLATE_ID")
        for probe_sampleset in self._cached_probes:
            probe_sampleset.references = list(
                grouped_matches.get_group(probe_sampleset.reference_id)[
                    "REFERENCE_TEMPLATE_ID"
                ]
            )

        return self._cached_probes

    def references(self, group="dev"):
        self._check_group(group)
        if self._cached_references is None:
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
        return ["1:1"]

    def _check_protocol(self, protocol):
        assert protocol in self.protocols(), "Unvalid protocol `{}` not in {}".format(
            protocol, self.protocols()
        )

    def _check_group(self, group):
        assert group in self.groups(), "Unvalid group `{}` not in {}".format(
            group, self.groups()
        )
