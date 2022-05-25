import copy
import logging
import os

from functools import partial

import numpy as np

import bob.io.base

from bob.bio.base.pipelines.abstract_classes import Database
from bob.extension import rc
from bob.extension.download import get_file
from bob.pipelines.sample import DelayedSample, SampleSet

logger = logging.getLogger("bob.bio.face")


class RFWDatabase(Database):
    """
    Dataset interface for the Racial faces in the wild dataset:

    The RFW is a subset of the MS-Celeb 1M dataset, and it's composed of 44332 images split into 11416 identities.
    There are four "race" labels in this dataset (`African`, `Asian`, `Caucasian`, and `Indian`).
    Furthermore, with the help of https://query.wikidata.org/ we've added information about gender and
    country of birth.

    We offer two evaluation protocols.
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
            score_all_vs_all=False,
            annotation_type="eyes-center",
            fixed_positions=None,
            memory_demanding=False,
        )

        self._pairs = dict()
        self._first_reference_of_subject = (
            dict()
        )  # Used with the Idiap protocol
        self._inverted_pairs = dict()
        self._id_race = dict()  # ID -- > RACE
        self._race_ids = dict()  # RACE --> ID
        self._landmarks = dict()
        self._cached_biometric_references = None
        self._cached_probes = None
        self._cached_zprobes = None
        self._cached_treferences = None
        self._cached_treferences = None
        self._discarded_subjects = (
            []
        )  # Some subjects were labeled with both races
        self._load_metadata(target_set="test")
        self._demographics = None
        self._demographics = self._get_demographics_dict()

        # Setting the seed for the IDIAP PROTOCOL,
        # so we have a consisent set of probes
        self._idiap_protocol_seed = 652

        # Number of samples used to Z-Norm and T-Norm (per race)
        self._nzprobes = 25
        self._ntreferences = 25

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/msceleb_wikidata_demographics.csv.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/msceleb_wikidata_demographics.csv.tar.gz",
        ]

    def _get_demographics_dict(self):
        """
        Get the dictionary with GENDER and COUNTRY of birth.
        Data obtained using the wiki data `https://query.wikidata.org/` using the following sparql query

        '''
        SELECT ?item ?itemLabel ?genderLabel ?countryLabel WHERE {
        ?item wdt:P31 wd:Q5.
        ?item ?label "{MY_NAME_HERE}"@en .
        optional{ ?item wdt:P21 ?gender.}
        optional{ ?item wdt:P27 ?country.}
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        '''


        """

        urls = RFWDatabase.urls()
        filename = get_file(
            "msceleb_wikidata_demographics.csv.tar.gz",
            urls,
            file_hash="8eb0e3c93647dfa0c13fade5db96d73a",
            extract=True,
        )[:-7]
        if self._demographics is None:
            self._demographics = dict()
            with open(filename) as f:
                for line in f.readlines():
                    line = line.split(",")
                    self._demographics[line[0]] = [
                        line[2],
                        line[3].rstrip("\n"),
                    ]

        return self._demographics

    def _get_subject_from_key(self, key):
        return key[:-5]

    def _load_metadata(self, target_set="test"):
        for race in self._races:

            pair_file = os.path.join(
                self.original_directory,
                target_set,
                "txts",
                race,
                f"{race}_pairs.txt",
            )

            for line in open(pair_file).readlines():
                line = line.split("\t")
                line[-1] = line[-1].rstrip("\n")

                key = f"{line[0]}_000{line[1]}"
                subject_id = self._get_subject_from_key(key)
                dict_key = f"{race}/{subject_id}/{key}"

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
                if len(line) == 3:
                    k_value = f"{line[0]}_000{line[2]}"
                    dict_value = f"{race}/{self._get_subject_from_key(k_value)}/{k_value}"
                else:
                    k_value = f"{line[2]}_000{line[3]}"
                    dict_value = f"{race}/{self._get_subject_from_key(k_value)}/{k_value}"

                if dict_key not in self._pairs:
                    self._pairs[dict_key] = []
                self._pairs[dict_key].append(dict_value)

        # Picking the first reference
        if self.protocol == "idiap":
            for p in self._pairs:
                _, subject_id, reference_id = p.split("/")
                if subject_id in self._first_reference_of_subject:
                    continue
                self._first_reference_of_subject[subject_id] = reference_id

        # Preparing the probes
        self._inverted_pairs = self._invert_dict(self._pairs)
        self._race_ids = self._invert_dict(self._id_race)

    def _invert_dict(self, dict_pairs):
        inverted_pairs = dict()

        for k in dict_pairs:
            if isinstance(dict_pairs[k], list):
                for v in dict_pairs[k]:
                    if v not in inverted_pairs:
                        inverted_pairs[v] = []
                    inverted_pairs[v].append(k)
            else:
                v = dict_pairs[k]
                if v not in inverted_pairs:
                    inverted_pairs[v] = []
                inverted_pairs[v].append(k)
        return inverted_pairs

    def background_model_samples(self):
        return []

    def _get_zt_samples(self, seed):

        cache = []

        # Setting the seed for the IDIAP PROTOCOL,
        # so we have a consisent set of probes
        np.random.seed(seed)

        for race in self._races:
            data_dir = os.path.join(
                self.original_directory, "train", "data", race
            )
            files = os.listdir(data_dir)
            # SHUFFLING
            np.random.shuffle(files)
            files = files[0 : self._nzprobes]

            # RFW original data is not super organized
            # train data from Caucasians are stored differently
            if race == "Caucasian":
                for f in files:
                    reference_id = os.listdir(os.path.join(data_dir, f))[0]
                    key = f"{race}/{f}/{reference_id[:-4]}"
                    cache.append(
                        self._make_sampleset(
                            key, target_set="train", get_demographic=False
                        )
                    )

            else:
                for f in files:
                    key = f"{race}/{race}/{f[:-4]}"
                    cache.append(
                        self._make_sampleset(
                            key, target_set="train", get_demographic=False
                        )
                    )
        return cache

    def zprobes(self, group="dev", proportion=1.0):
        if self._cached_zprobes is None:
            self._cached_zprobes = self._get_zt_samples(
                self._idiap_protocol_seed + 1
            )
            references = list(
                set([s.reference_id for s in self.references(group=group)])
            )
            for p in self._cached_zprobes:
                p.references = copy.deepcopy(references)

        return self._cached_zprobes

    def treferences(self, group="dev", proportion=1.0):
        if self._cached_treferences is None:
            self._cached_treferences = self._get_zt_samples(
                self._idiap_protocol_seed + 2
            )

        return self._cached_zprobes

    def probes(self, group="dev"):
        self._check_group(group)
        if self._cached_probes is None:

            # Setting the seed for the IDIAP PROTOCOL,
            # so we have a consisent set of probes
            np.random.seed(self._idiap_protocol_seed)

            self._cached_probes = []
            for key in self._inverted_pairs:
                sset = self._make_sampleset(key)
                sset.references = [
                    key.split("/")[-1] for key in self._inverted_pairs[key]
                ]

                # If it's the idiap protocol, we should
                # extend the list of comparisons
                if self.protocol == "idiap":
                    # Picking one reference per race
                    extra_references = []
                    for k in self._race_ids:
                        # Discard samples from the same race
                        if k == sset.race:
                            continue

                        index = np.random.randint(len(self._race_ids[k]))
                        random_subject_id = self._race_ids[k][index]

                        # Search for the first reference id in with this identity
                        extra_references.append(
                            self._first_reference_of_subject[random_subject_id]
                        )

                    assert len(extra_references) == 3

                    sset.references += extra_references

                self._cached_probes.append(sset)
        return self._cached_probes

    def _fetch_landmarks(self, filename, key):

        if key not in self._landmarks:
            with open(filename) as f:
                for line in f.readlines():
                    line = line.split("\t")
                    # pattern 'm.0c7mh2_0003.jpg'[:-4]
                    k = line[0].split("/")[-1][:-4]
                    self._landmarks[k] = dict()
                    self._landmarks[k]["reye"] = (
                        float(line[3]),
                        float(line[2]),
                    )
                    self._landmarks[k]["leye"] = (
                        float(line[5]),
                        float(line[4]),
                    )

        return self._landmarks[key]

    def _make_sampleset(self, item, target_set="test", get_demographic=True):
        race, subject_id, reference_id = item.split("/")

        # RFW original data is not super organized
        # Test and train data os stored differently

        key = f"{race}/{subject_id}/{reference_id}"

        path = (
            os.path.join(
                self.original_directory,
                f"{target_set}/data/{race}",
                subject_id,
                reference_id + self._default_extension,
            )
            if (target_set == "test" or race == "Caucasian")
            else os.path.join(
                self.original_directory,
                f"{target_set}/data/{race}",
                reference_id + self._default_extension,
            )
        )

        annotations = (
            self._fetch_landmarks(
                os.path.join(
                    self.original_directory, "erratum1", "Caucasian_lmk.txt"
                ),
                reference_id,
            )
            if (target_set == "train" and race == "Caucasian")
            else self._fetch_landmarks(
                os.path.join(
                    self.original_directory,
                    f"{target_set}/txts/{race}/{race}_lmk.txt",
                ),
                reference_id,
            )
        )

        samples = [
            DelayedSample(
                partial(
                    bob.io.base.load,
                    path,
                ),
                key=key,
                annotations=annotations,
                reference_id=reference_id,
                subject_id=subject_id,
            )
        ]

        if get_demographic:
            gender = self._demographics[subject_id][0]
            country = self._demographics[subject_id][1]

            return SampleSet(
                samples,
                key=key,
                reference_id=reference_id,
                subject_id=subject_id,
                race=race,
                gender=gender,
                country=country,
            )
        else:
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
                self._cached_biometric_references.append(
                    self._make_sampleset(key)
                )

        return self._cached_biometric_references

    def all_samples(self, group="dev"):
        self._check_group(group)

        return self.references() + self.probes()

    def groups(self):
        return ["dev"]

    def protocols(self):
        return ["original", "idiap"]

    def _check_protocol(self, protocol):
        assert (
            protocol in self.protocols()
        ), "Unvalid protocol `{}` not in {}".format(protocol, self.protocols())

    def _check_group(self, group):
        assert group in self.groups(), "Unvalid group `{}` not in {}".format(
            group, self.groups()
        )
