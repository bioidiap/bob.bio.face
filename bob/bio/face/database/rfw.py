from bob.bio.base.pipelines.vanilla_biometrics.abstract_classes import Database
import pandas as pd
from bob.pipelines.sample import DelayedSample, SampleSet
from bob.extension import rc
import os
import bob.io.image
from functools import partial
import logging

logger = logging.getLogger("bob.bio.face")


class RFWDatabase(Database):
    """
    Dataset interface for the Racial faces in the wild dataset:

    The RFW is a subset of the MS-Celeb 1M dataset, and it's composed of 44332 images split into 11416 identities.
    There are four "race" labels in this dataset (`African`, `Asian`, `Caucasian`, and `Indian`).

    About the evaluation protocols, we offer two protocols.
    The first one, called "original" is the original protocol from its publication. It contains ~24k comparisons in total.
    Worth noting that this evaluation protocol has an issue. It considers only comparisons of pairs of images from the same
    "race".
    To close this gap, we've created a protocol called "idiap" that extends the original protocol to one where impostors  comparisons
    (or non-mated) is possible. This is closed to a real-world scenario.

    .. warning::
        The following identities are assossiated with two races in the original dataset
         - m.023915
         - m.0z08d8y
         - m.0bk56n
         - m.04f4wpb
         - m.0gc2xf9
         - m.08dyjb
         - m.05y2fd
         - m.0gbz836
         - m.01pw5d
         - m.0cm83zb
         - m.02qmpkk
         - m.05xpnv


    For more information check:

    .. code-block:: latex

        @inproceedings{wang2019racial,
        title={Racial faces in the wild: Reducing racial bias by information maximization adaptation network},
        author={Wang, Mei and Deng, Weihong and Hu, Jiani and Tao, Xunqiang and Huang, Yaohai},
        booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
        pages={692--702},
        year={2019}
        }

    """

    def __init__(
        self,
        protocol,
        original_directory=rc.get("bob.bio.face.rfw.directory"),
        **kwargs,
    ):

        if original_directory is None or not os.path.exists(original_directory):
            raise ValueError(
                "Invalid or non existant `original_directory`: f{original_directory}"
            )

        self._check_protocol(protocol)
        self._races = ["African", "Asian", "Caucasian", "Indian"]
        self.original_directory = original_directory
        self._default_extension = ".jpg"

        super().__init__(
            name="rfw",
            protocol=protocol,
            allow_scoring_with_all_biometric_references=False,
            annotation_type="eyes-center",
            fixed_positions=None,
            memory_demanding=False,
        )

        self._pairs = dict()
        self._inverted_pairs = dict()
        self._id_race = dict()
        self._landmarks = dict()
        self._cached_biometric_references = None
        self._cached_probes = None
        self._discarded_subjects = []  # Some subjects were labeled with both races
        self._load_metadata(target_set="test")

    def _get_subject_from_key(self, key):
        return key[:-5]
        # return key.split("/")[0]

    def _load_metadata(self, target_set="test"):
        for race in self._races:

            pair_file = os.path.join(
                self.original_directory, target_set, "txts", race, f"{race}_pairs.txt"
            )

            for l in open(pair_file).readlines():
                l = l.split("\t")
                l[-1] = l[-1].rstrip("\n")

                key = f"{l[0]}_000{l[1]}"
                subject_id = self._get_subject_from_key(key)
                dict_key = f"{race}/{subject_id}/{key}"
                # subject_id = self._get_subject_from_key(key)
                # subject_id = key[:-5]

                if subject_id not in self._id_race:
                    self._id_race[subject_id] = race
                else:
                    if (
                        self._id_race[subject_id] != race
                        and subject_id not in self._discarded_subjects
                    ):
                        logger.warning(
                            f"{subject_id} was already labeled as {self._id_race[subject_id]}, and it's illogical to be relabeled as {race}. "
                            f"This seems a problem with RFW dataset, so we are removing all samples linking {subject_id} as {race}"
                        )
                        self._discarded_subjects.append(subject_id)
                        continue

                # Positive or negative pairs
                if len(l) == 3:
                    # k = f"{l[0]}_000{l[2]}"
                    # value = f"{self._get_subject_from_key(key)}/{k}"
                    k_value = f"{l[0]}_000{l[2]}"
                    dict_value = (
                        f"{race}/{self._get_subject_from_key(k_value)}/{k_value}"
                    )
                else:
                    # k = f"{l[2]}_000{l[3]}"
                    # value = f"{self._get_subject_from_key(key)}/{k}"
                    k_value = f"{l[2]}_000{l[3]}"
                    dict_value = (
                        f"{race}/{self._get_subject_from_key(k_value)}/{k_value}"
                    )

                if dict_key not in self._pairs:
                    self._pairs[dict_key] = []
                self._pairs[dict_key].append(dict_value)

        # Preparing the probes
        self._invert_dict()

        pass

    def _invert_dict(self):
        self._inverted_pairs = dict()

        for k in self._pairs:
            for v in self._pairs[k]:
                if v not in self._inverted_pairs:
                    self._inverted_pairs[v] = []
                self._inverted_pairs[v].append(k)

    def background_model_samples(self):
        return None

    def probes(self, group="dev"):
        self._check_group(group)
        if self._cached_probes is None:

            self._cached_probes = []
            for key in self._inverted_pairs:
                sset = self._make_sampleset(key)
                sset.references = [
                    key.split("/")[-1] for key in self._inverted_pairs[key]
                ]
                self._cached_probes.append(sset)

        return self._cached_probes

    def _fetch_landmarks(self, filename, key):

        if key not in self._landmarks:
            with open(filename) as f:
                for line in f.readlines():
                    line = line.split("\t")
                    # pattern 'm.0c7mh2_0003.jpg'[:-4]
                    key = line[0].split("/")[-1][:-4]
                    self._landmarks[key] = dict()
                    self._landmarks[key]["reye"] = (float(line[3]), float(line[2]))
                    self._landmarks[key]["leye"] = (float(line[5]), float(line[4]))

        return self._landmarks[key]

    def _make_sampleset(self, item, target_set="test"):
        race, subject_id, reference_id = item.split("/")

        key = f"{race}/{subject_id}/{reference_id}"

        samples = [
            DelayedSample(
                partial(
                    bob.io.image.load,
                    os.path.join(
                        self.original_directory,
                        f"{target_set}/data/{race}",
                        subject_id,
                        reference_id + self._default_extension,
                    ),
                ),
                key=key,
                annotations=self._fetch_landmarks(
                    os.path.join(
                        self.original_directory,
                        f"{target_set}/txts/{race}/{race}_lmk.txt",
                    ),
                    reference_id,
                ),
            )
        ]

        return SampleSet(
            samples,
            key=key,
            reference_id=reference_id,
            subject_id=subject_id,
            race=race,
        )

    def references(self, group="dev"):
        self._check_group(group)
        if self._cached_biometric_references is None:
            self._cached_biometric_references = []
            for key in self._pairs:
                self._cached_biometric_references.append(self._make_sampleset(key))

        return self._cached_biometric_references

    def all_samples(self, group="dev"):
        self._check_group(group)

        return self.references() + self.probes()

    def groups(self):
        return ["dev"]

    def protocols(self):
        return ["original", "diap"]

    def _check_protocol(self, protocol):
        assert protocol in self.protocols(), "Unvalid protocol `{}` not in {}".format(
            protocol, self.protocols()
        )

    def _check_group(self, group):
        assert group in self.groups(), "Unvalid group `{}` not in {}".format(
            group, self.groups()
        )
