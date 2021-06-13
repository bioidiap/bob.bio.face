from bob.bio.base.pipelines.vanilla_biometrics.abstract_classes import Database
import pandas as pd
from bob.pipelines.sample import DelayedSample, SampleSet
from bob.extension import rc
import os
import bob.io.image
from functools import partial


def load(path):
    return bob.io.image.load(os.path.join(rc["bob.db.ijbc.directory"], path))


def _make_sample_from_template_row(row, image_directory):
    return DelayedSample(
        load=partial(
            bob.io.image.load, path=os.path.join(image_directory, row["FILENAME"])
        ),
        template_id=str(row["TEMPLATE_ID"]),
        subject_id=str(row["SUBJECT_ID"]),
        key=os.path.splitext(row["FILENAME"])[0],
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
        samples, template_id=samples[0].template_id, subject_id=samples[0].subject_id
    )


class IJBCDatabase(Database):
    def __init__(
        self,
        protocol="1:1",
        original_directory=rc["bob.bio.face.ijbc.directory"],
        **kwargs
    ):
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

        self._load_metadata()

    def _load_metadata(self):
        # Load CSV files
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
                grouped_matches.get_group(int(probe_sampleset.template_id))[
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
