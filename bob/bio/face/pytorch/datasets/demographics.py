#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
Datasets that handles demographic information

"""


import itertools
import logging
import os
import random

import cloudpickle
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

import bob.io.base

from bob.bio.face.database import (
    MEDSDatabase,
    MobioDatabase,
    MorphDatabase,
    RFWDatabase,
    VGG2Database,
)
from bob.extension import rc
from bob.extension.download import get_file

logger = logging.getLogger(__name__)


class DemographicTorchDataset(Dataset):
    """
    Pytorch base dataset that handles demographic information

    Parameters
    ----------

        bob_dataset:
          Instance of a bob database object

        transform=None

    """

    def __init__(self, bob_dataset, transform=None):

        self.bob_dataset = bob_dataset
        self.transform = transform
        self.load_bucket()

    def __len__(self):
        return len(self.bucket)

    @property
    def n_classes(self):
        return len(self.labels)

    @property
    def n_samples(self):
        return len(self.bucket)

    @property
    def demographic_keys(self):
        return self._demographic_keys

    def __getitem__(self, idx):
        """
        It dumps a dictionary containing the following keys: data, label, demography

        """

        sample = self.bucket[idx]

        image = (
            sample.data
            if self.transform is None
            else self.transform(sample.data)
        )

        # image = image.astype("float32")

        label = self.labels[sample.subject_id]

        demography = self.get_demographics(sample)

        return {"data": image, "label": label, "demography": demography}

    def count_subjects_per_demographics(self):
        """
        Count the number of subjects per demographics
        """
        all_demographics = list(self.subject_demographic.values())

        # Number of subjects per demographic
        subjects_per_demographics = dict(
            [
                (d, sum(np.array(all_demographics) == d))
                for d in set(all_demographics)
            ]
        )

        return subjects_per_demographics

    def get_demographic_weights(self, as_dict=True):
        """
        Compute the inverse weighting for each demographic group.


        .. warning::
           This is not the same function as `get_demographic_class_weights`.

        Parameters
        ----------
           If `True` will return the weights as a dict.

        """
        n_identities = len(self.subject_demographic)

        # Number of subjects per demographic
        subjects_per_demographics = self.count_subjects_per_demographics()

        # INverse probability (1-p_i)/p_i
        demographic_weights = dict()
        for i in subjects_per_demographics:
            p_i = subjects_per_demographics[i] / n_identities
            demographic_weights[i] = (1 - p_i) / p_i

        p_accumulator = sum(demographic_weights.values())
        # Scaling the inverse probability
        for i in demographic_weights:
            demographic_weights[i] /= p_accumulator

        # Return as a dictionary
        if as_dict:
            return demographic_weights

        # Returning as a list (this is more aproppriated for NN training)
        return [demographic_weights[k] for k in self.demographic_keys]

    def get_demographic_class_weights(self):
        """
        Compute the class weights based on the demographics

        Returns
        -------
            weights: list
              A list containing the weights for each class
        """

        subjects_per_demographics = self.count_subjects_per_demographics()
        demographic_weights = self.get_demographic_weights()

        weights = [
            demographic_weights[v] / subjects_per_demographics[v]
            for k, v in self.subject_demographic.items()
        ]

        return torch.Tensor(weights)


class MedsTorchDataset(DemographicTorchDataset):
    """
    MEDS torch interface

    .. warning::
       Unfortunatelly, in this dataset there are several identities that has only ONE sample.
       Hence, it is impossible to properly use this dataset to do contrastive learning, for instance.
       If this is thecase, please set `take_from_znorm=True`, so the `dev` or the `eval` sets are used.


    Parameters
    ----------

    protocol: str
        One of the MEDS available protocols, check :py:class:`bob.bio.face.database.MEDSDatabase`

    database_path: str
        Database path

    database_extension: str
        Database extension

    transform: callable
       Transformation function to the input sample

    take_from_znorm: bool
       If `True`, it will take the samples from `treferences` and `zprobes` methods that comes from the training set
       If `False`, it will take the samples from `references` and `probes` methods. Then, the variable `group` is considered.

    group: str

    """

    def __init__(
        self,
        protocol,
        database_path,
        database_extension=".h5",
        transform=None,
        take_from_znorm=False,
        group="dev",
    ):

        bob_dataset = MEDSDatabase(
            protocol=protocol,
            dataset_original_directory=database_path,
            dataset_original_extension=database_extension,
        )
        self.take_from_znorm = take_from_znorm
        self.group = group
        super().__init__(bob_dataset, transform=transform)

    def load_bucket(self):
        self._target_metadata = "rac"

        if self.take_from_znorm:
            self.bucket = [
                s for sset in self.bob_dataset.zprobes() for s in sset
            ]
            self.bucket += [
                s for sset in self.bob_dataset.treferences() for s in sset
            ]
        else:
            self.bucket = [
                s
                for sset in self.bob_dataset.probes(group=self.group)
                for s in sset
            ]
            self.bucket += [
                s
                for sset in self.bob_dataset.references(group=self.group)
                for s in sset
            ]

        offset = 0
        self.labels = dict()
        self.subject_demographic = dict()

        for s in self.bucket:
            if s.subject_id not in self.labels:
                self.labels[s.subject_id] = offset
                self.subject_demographic[s.subject_id] = getattr(
                    s, self._target_metadata
                )
                offset += 1

        metadata_keys = set(self.subject_demographic.values())
        self._demographic_keys = dict(
            zip(metadata_keys, range(len(metadata_keys)))
        )

    def get_demographics(self, sample):
        demographic_key = getattr(sample, "rac")
        return self._demographic_keys[demographic_key]


class VGG2TorchDataset(DemographicTorchDataset):
    """
    VGG2 for torch.

    This interface make usage of :any:`bob.bio.face.database.VGG2Database`.

    The "race" labels below were annotated by the students from the period 2018-2020. Race labels taken from: MasterEBTSv10.0.809302017_Final.pdf

    - A: Asian in general (Chinese, Japanese, Filipino, Korean, Polynesian, Indonesian, Samoan, or any other Pacific Islander
    - B: A person having origins in any of the black racial groups of Africa
    - I: American Indian, Asian Indian, Eskimo, or Alaskan native
    - U: Of indeterminable race
    - W: Caucasian, Mexican, Puerto Rican, Cuban, Central or South American, or other Spanish culture or origin, Regardless of race
    - N: None of the above


    Gender information was taken from the original dataset
    There are the following genders available:
     - male
     - female


    .. note::
        Some important information about this interface.
        We have the following statistics:
            - n_classes = 8631
            - n_demographics: 12 ['m-A': 0, 'm-B': 1, 'm-I': 2, 'm-U': 3, 'm-W': 4, 'm-N': 5, 'f-A': 6, 'f-B': 7, 'f-I': 8, 'f-U': 9, 'f-W': 10, 'f-N': 11]


    .. note::

       Follow the distribution the combination of race and gender demographics
       {'m-B': 552, 'm-U': 64, 'm-W': 3903, 'f-W': 2657, 'f-A': 286, 'f-U': 34, 'f-I': 298, 'f-N': 2, 'f-B': 200, 'm-N': 1, 'm-I': 366, 'm-A': 268}

       Note that `m-N` has 1 subject and 'f-N' has 2 subjects.
       For this reason, we are removing this race from this interface.
       We can't learn anything from one sample.


    Parameters
    ----------
        database_path: str
           Path containing the raw data

        database_extension:

        load_bucket_from_cache: bool
          If set, it will load the list of available samples from the cache

        train: bool
          If set it will prepare a bucket for training.

        include_u_n: bool
          If `True` it will include 'U' (Undefined) and 'N' (None) on the list of races.


    """

    def __init__(
        self,
        protocol,
        database_path,
        database_extension=".jpg",
        transform=None,
        load_bucket_from_cache=True,
        include_u_n=False,
        train=True,
    ):

        bob_dataset = VGG2Database(
            protocol=protocol,
            dataset_original_directory=database_path,
            dataset_original_extension=database_extension,
        )
        self.load_bucket_from_cache = load_bucket_from_cache

        # Percentage of the samples used for training
        self._percentage_for_training = 0.8
        self.train = train

        # All possible metadata
        self._possible_genders = ["m", "f"]

        # self._possible_races = ["A", "B", "I", "U", "W", "N"]
        self._possible_races = ["A", "B", "I", "W"]
        if include_u_n:
            self._possible_races += ["U", "N"]

        super().__init__(bob_dataset, transform=transform)

    def decode_race(self, race):
        # return race if race in self._possible_races else "N"
        return race if race in self._possible_races else "W"

    def get_key(self, sample):
        return f"{sample.gender}-{self.decode_race(sample.race)}"

    def get_cache_path(self):

        filename = (
            "vgg2_short_cached_bucket.pickle"
            if self.bob_dataset.protocol == "vgg2-short"
            else "vgg2_full_cached_bucket.pickle"
        )

        return os.path.join(
            rc.get(
                "bob_data_folder",
                os.path.join(os.path.expanduser("~"), "bob_data"),
            ),
            "datasets",
            f"{filename}",
        )

    def cache_bucket(self, bucket):
        """
        Cache the list of samples into a temporary directory
        """
        bucket_filename = self.get_cache_path()
        os.makedirs(os.path.dirname(bucket_filename), exist_ok=True)
        with open(bucket_filename, "wb") as f:
            cloudpickle.dump(bucket, f)

    def load_cached_bucket(self):
        bucket_filename = self.get_cache_path()
        with open(bucket_filename, "rb") as f:
            bucket = cloudpickle.load(f)
        return bucket

    def load_bucket(self):

        # Defining the demographics keys
        self._demographic_keys = [
            f"{gender}-{race}"
            for gender in self._possible_genders
            for race in self._possible_races
        ]
        self._demographic_keys = dict(
            [(d, i) for i, d in enumerate(self._demographic_keys)]
        )

        # Loading the buket from cache
        if self.load_bucket_from_cache and os.path.exists(
            self.get_cache_path()
        ):
            self.bucket = self.load_cached_bucket()
        else:
            self.bucket = [
                s for s in self.bob_dataset.background_model_samples()
            ]
            # Caching the bucket
            self.cache_bucket(self.bucket)

        # Mapping subject_id with labels
        self.labels = sorted(list(set([s.subject_id for s in self.bucket])))
        self.labels = dict([(l, i) for i, l in enumerate(self.labels)])

        # Spliting the bucket into training and developement set
        all_indexes = np.array([self.labels[x.subject_id] for x in self.bucket])
        indexes = []
        if self.train:
            for i in range(self.n_classes):
                ind = np.where(all_indexes == i)[0]
                indexes += list(
                    ind[
                        0 : int(
                            np.floor(len(ind) * self._percentage_for_training)
                        )
                    ]
                )
        else:
            for i in range(self.n_classes):
                ind = np.where(all_indexes == i)[0]
                indexes += list(
                    ind[
                        int(
                            np.floor(len(ind) * self._percentage_for_training)
                        ) :
                    ]
                )

        # Redefining the bucket
        self.bucket = list(np.array(self.bucket)[indexes])

        # Mapping subject and demographics for fast access
        self.subject_demographic = dict()

        for s in self.bucket:
            if s.subject_id not in self.subject_demographic:
                self.subject_demographic[s.subject_id] = self.get_key(s)

    def get_demographics(self, sample):
        demographic_key = self.get_key(sample)
        return self._demographic_keys[demographic_key]


class MorphTorchDataset(DemographicTorchDataset):
    """
    MORPH torch interface

    .. warning::
       Unfortunatelly, in this dataset there are several identities that has only ONE sample.
       Hence, it is impossible to properly use this dataset to do contrastive learning, for instance.
       If this is thecase, please set `take_from_znorm=True`, so the `dev` or the `eval` sets are used.


    Parameters
    ----------

    protocol: str
        One of the Morph available protocols, check :py:class:`bob.bio.face.database.MEDSDatabase`

    database_path: str
        Database path

    database_extension: str
        Database extension

    transform: callable
       Transformation function to the input sample

    take_from_znorm: bool
       If `True`, it will take the samples from `treferences` and `zprobes` methods that comes from the training set
       If `False`, it will take the samples from `references` and `probes` methods. Then, the variable `group` is considered.

    group: str

    """

    def __init__(
        self,
        protocol,
        database_path,
        database_extension=".h5",
        transform=None,
        take_from_znorm=True,
        group="dev",
    ):

        bob_dataset = MorphDatabase(
            protocol=protocol,
            dataset_original_directory=database_path,
            dataset_original_extension=database_extension,
        )
        self.take_from_znorm = take_from_znorm
        self.group = group

        super().__init__(bob_dataset, transform=transform)

    def load_bucket(self):

        # Morph dataset has an intersection in between zprobes and treferences
        # Those are the
        self.excluding_list = [
            "190276",
            "332158",
            "111942",
            "308129",
            "334074",
            "350814",
            "131677",
            "168724",
            "276055",
            "275589",
            "286810",
        ]

        if self.take_from_znorm:
            self.bucket = [
                s for sset in self.bob_dataset.zprobes() for s in sset
            ]
            self.bucket += [
                s
                for sset in self.bob_dataset.treferences()
                for s in sset
                if sset.subject_id not in self.excluding_list
            ]
        else:
            self.bucket = [
                s
                for sset in self.bob_dataset.probes(group=self.group)
                for s in sset
            ]
            self.bucket += [
                s
                for sset in self.bob_dataset.references(group=self.group)
                for s in sset
            ]

        offset = 0
        self.labels = dict()
        self.subject_demographic = dict()

        for s in self.bucket:
            if s.subject_id not in self.labels:
                self.labels[s.subject_id] = offset
                self.subject_demographic[s.subject_id] = f"{s.rac}-{s.sex}"
                offset += 1

        metadata_keys = set(self.subject_demographic.values())
        self._demographic_keys = dict(
            zip(metadata_keys, range(len(metadata_keys)))
        )

    def get_demographics(self, sample):
        demographic_key = f"{sample.rac}-{sample.sex}"
        return self._demographic_keys[demographic_key]


class RFWTorchDataset(DemographicTorchDataset):
    def __init__(
        self, protocol, database_path, database_extension=".h5", transform=None
    ):

        bob_dataset = RFWDatabase(
            protocol=protocol,
            dataset_original_directory=database_path,
            dataset_original_extension=database_extension,
        )
        super().__init__(bob_dataset, transform=transform)

    def load_demographics(self):

        target_metadata = "race"
        metadata_keys = set(
            [
                getattr(sset, target_metadata)
                for sset in self.bob_dataset.zprobes()
            ]
            + [
                getattr(sset, target_metadata)
                for sset in self.bob_dataset.treferences()
            ]
        )
        metadata_keys = dict(zip(metadata_keys, range(len(metadata_keys))))
        return metadata_keys

    def get_demographics(self, sample):
        demographic_key = getattr(sample, "race")
        return self._demographic_keys[demographic_key]


class MobioTorchDataset(DemographicTorchDataset):
    def __init__(
        self, protocol, database_path, database_extension=".h5", transform=None
    ):
        bob_dataset = MobioDatabase(
            protocol=protocol,
            dataset_original_directory=database_path,
            dataset_original_extension=database_extension,
        )

        super().__init__(bob_dataset, transform=transform)

    def load_bucket(self):
        self._target_metadata = "gender"
        self.bucket = [s for s in self.bob_dataset.background_model_samples()]
        offset = 0
        self.labels = dict()
        self.subject_demographic = dict()

        for s in self.bucket:
            if s.subject_id not in self.labels:
                self.labels[s.subject_id] = offset
                self.subject_demographic[s.subject_id] = getattr(
                    s, self._target_metadata
                )
                offset += 1

        metadata_keys = set(self.subject_demographic.values())
        self._demographic_keys = dict(
            zip(metadata_keys, range(len(metadata_keys)))
        )

    def __len__(self):
        return len(self.bucket)

    def get_demographics(self, sample):
        demographic_key = getattr(sample, self._target_metadata)
        return self._demographic_keys[demographic_key]


class MSCelebTorchDataset(DemographicTorchDataset):
    """
    This interface make usage of a CSV file containing gender and
    RACE annotations available at.

    The "race" labels below were annotated by the students from the period 2018-2020. Race labels taken from: MasterEBTSv10.0.809302017_Final.pdf

    - A: Asian in general (Chinese, Japanese, Filipino, Korean, Polynesian, Indonesian, Samoan, or any other Pacific Islander
    - B: A person having origins in any of the black racial groups of Africa
    - I: American Indian, Asian Indian, Eskimo, or Alaskan native
    - U: Of indeterminable race
    - W: Caucasian, Mexican, Puerto Rican, Cuban, Central or South American, or other Spanish culture or origin, Regardless of race
    - N: None of the above


    Gender and country information taken from the wiki data: https://www.wikidata.org/wiki/Wikidata:Main_Page
    There are the following genders available:
     - male
     - female
     - other


    .. note::
        Some important information about this interface.
        If `include_unknow_demographics==False` we will have the following statistics:
            - n_classes = 81279
            - n_demographics: 15 ['male-A', 'male-B', 'male-I', 'male-U', 'male-W', 'female-A', 'female-B', 'female-I', 'female-U', 'female-W', 'other-A', 'other-B', 'other-I', 'other-U', 'other-W']


        If `include_unknow_demographics==True` we will have the following statistics:
            - n_classes = 89735
            - n_demographics: 18 ['male-A', 'male-B', 'male-I', 'male-N', 'male-U', 'male-W', 'female-A', 'female-B', 'female-I', 'female-N', 'female-U', 'female-W', 'other-A', 'other-B', 'other-I', 'other-N', 'other-U', 'other-W']



    Parameters
    ----------
        database_path: str
           Path containing the raw data

        database_extension:

        idiap_path: bool
          If set, it will use the idiap standard relative path to load the data (e.g. [BASE_PATH]/chunk_[n]/[user_id])

        include_unknow_demographics: bool
          If set, it will include subjects whose race was set to `N` (None of the above)

        load_bucket_from_cache: bool
          If set, it will load the list of available samples from the cache


    """

    def __init__(
        self,
        database_path,
        database_extension=".png",
        idiap_path=True,
        include_unknow_demographics=False,
        load_bucket_from_cache=True,
        transform=None,
    ):
        self.idiap_path = idiap_path
        self.database_path = database_path
        self.database_extension = database_extension
        self.include_unknow_demographics = include_unknow_demographics
        self.load_bucket_from_cache = load_bucket_from_cache
        self.transform = transform

        # Private keys
        self._possible_genders = ["male", "female", "other"]

        # filename = "/idiap/user/tpereira/gitlab/bob/database-purgatory/wikidata/msceleb_race_wikidata.csv"
        urls = MSCelebTorchDataset.urls()
        filename = (
            get_file(
                "msceleb_race_wikidata.tar.gz",
                urls,
                file_hash="76339d73f352faa00c155f7040e772bb",
                extract=True,
            )[:-7]
            + ".csv"
        )

        self.load_bucket(filename)

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/msceleb_race_wikidata.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/msceleb_race_wikidata.tar.gz",
        ]

    def get_cache_path(self):

        filename = (
            "msceleb_cached_bucket_WITH_unknow_demographics.csv"
            if self.include_unknow_demographics
            else "msceleb_cached_bucket_WITHOUT_unknow_demographics.csv"
        )

        return os.path.join(
            rc.get(
                "bob_data_folder",
                os.path.join(os.path.expanduser("~"), "bob_data"),
            ),
            "datasets",
            f"{filename}",
        )

    def cache_bucket(self, bucket):
        """
        Cache the list of samples into a temporary directory
        """
        bucket_filename = self.get_cache_path()
        os.makedirs(os.path.dirname(bucket_filename), exist_ok=True)
        with open(bucket_filename, "w") as f:
            for b in bucket:
                f.write(f"{b}\n")

    def load_cached_bucket(self):
        """
        Load the bucket from the cache
        """
        bucket_filename = self.get_cache_path()
        return [f.rstrip("\n") for f in open(bucket_filename).readlines()]

    def __len__(self):
        return len(self.bucket)

    def load_bucket(self, csv_filename):

        dataframe = pd.read_csv(csv_filename)

        # Possible races
        # {'A', 'B', 'I', 'N', 'U', 'W', nan}

        filtered_dataframe = (
            dataframe.loc[
                (dataframe.RACE == "A")
                | (dataframe.RACE == "B")
                | (dataframe.RACE == "I")
                | (dataframe.RACE == "U")
                | (dataframe.RACE == "W")
                | (dataframe.RACE == "N")
            ]
            if self.include_unknow_demographics
            else dataframe.loc[
                (dataframe.RACE == "A")
                | (dataframe.RACE == "B")
                | (dataframe.RACE == "I")
                | (dataframe.RACE == "U")
                | (dataframe.RACE == "W")
            ]
        )

        filtered_dataframe_list = filtered_dataframe[
            ["idiap_chunk", "ID"]
        ].to_csv()

        # Defining the number of classes
        subject_relative_paths = [
            os.path.join(ll.split(",")[1], ll.split(",")[2])
            for ll in filtered_dataframe_list.split("\n")[1:-1]
        ]

        if self.load_bucket_from_cache and os.path.exists(
            self.get_cache_path()
        ):
            self.bucket = self.load_cached_bucket()
        else:
            # Defining all images
            logger.warning(
                f"Fetching all samples paths on the fly. This might take some minutes."
                f"Then this will be cached in {self.get_cache_path()} and loaded from this cache"
            )

            self.bucket = [
                os.path.join(subject, f)
                for subject in subject_relative_paths
                for f in os.listdir(os.path.join(self.database_path, subject))
                if f[-4:] == self.database_extension
            ]
            self.cache_bucket(self.bucket)

        self.labels = dict(
            [
                (k.split("/")[-1], i)
                for i, k in enumerate(subject_relative_paths)
            ]
        )

        # Setting the possible demographics and the demographic keys
        filtered_dataframe = filtered_dataframe.set_index("ID")
        self.metadata = filtered_dataframe[["GENDER", "RACE"]].to_dict(
            orient="index"
        )

        self._demographic_keys = [
            f"{gender}-{race}"
            for gender in self._possible_genders
            for race in sorted(set(filtered_dataframe["RACE"]))
        ]
        self._demographic_keys = dict(
            [(d, i) for i, d in enumerate(self._demographic_keys)]
        )

        # Creating a map between the subject and the demographic
        self.subject_demographic = dict(
            [(m, self.get_demographics(m)) for m in self.metadata]
        )

    def get_demographics(self, subject_id):
        race = self.metadata[subject_id]["RACE"]
        gender = self.metadata[subject_id]["GENDER"]

        gender = "other" if gender != "male" and gender != "female" else gender

        return self._demographic_keys[f"{gender}-{race}"]

    def __getitem__(self, idx):

        sample = self.bucket[idx]

        subject_id = sample.split("/")[-2]

        # Transforming the image
        image = bob.io.base.load(os.path.join(self.database_path, sample))

        image = image if self.transform is None else self.transform(image)

        label = self.labels[subject_id]

        # Getting the demographics

        demography = self.get_demographics(subject_id)

        return {"data": image, "label": label, "demography": demography}


class SiameseDemographicWrapper(Dataset):
    """
    This class wraps the current demographic interface and
    dumps random positive and negative pairs of samples

    """

    def __init__(
        self,
        demographic_dataset,
        max_positive_pairs_per_subject=20,
        negative_pairs_per_subject=3,
        dense_negatives=False,
    ):

        self.demographic_dataset = demographic_dataset
        self.max_positive_pairs_per_subject = max_positive_pairs_per_subject
        self.negative_pairs_per_subject = negative_pairs_per_subject

        # Creating a bucket mapping the items of the bucket with their respective identities
        self.siamese_bucket = dict()
        for b in demographic_dataset.bucket:
            if b.subject_id not in self.siamese_bucket:
                self.siamese_bucket[b.subject_id] = []

            self.siamese_bucket[b.subject_id].append(b)

        positive_pairs = self.create_positive_pairs()
        if dense_negatives:
            negative_pairs = self.create_dense_negative_pairs()
        else:
            negative_pairs = self.create_light_negative_pairs()

        # Redefining the bucket
        self.siamese_bucket = negative_pairs + positive_pairs

        self.labels = np.hstack(
            (np.zeros(len(negative_pairs)), np.ones(len(positive_pairs)))
        )

        pass

    def __len__(self):
        return len(self.siamese_bucket)

    def create_positive_pairs(self):

        # Creating positive pairs for each identity
        positives = []
        random.seed(0)
        for b in self.siamese_bucket:
            samples = self.siamese_bucket[b]
            random.shuffle(samples)

            # All possible pair combinations
            samples = itertools.combinations(samples, 2)

            positives += [
                s
                for s in list(samples)[0 : self.max_positive_pairs_per_subject]
            ]
            pass

        return positives

    def create_dense_negative_pairs(self):
        """
        Creating negative pairs.
        Here we create only negative pairs from the same demographic group,
        since we know that pairs from different demographics leads to
        poor scores


        .. warning:
           The list of negative pairs is dense.
           For each combination of subjects for a particular demographic,
           we will take `negative_pairs_per_subject` samples.
           Hence, the number of negative pairs can explode as a function
           of number of subjects.
           For example, a combination pairs with 1000 identities gives us
           499500 pairs. Taking `3` pairs of images for these combinations
           of identities will give us ~1.5M negative pairs.
           Hence, be careful with that.

        """

        # Inverting subject
        random.seed(0)
        negatives = []

        # Creating the dictionary containing the demographics--> subjects
        demographic_subject = dict()
        for k, v in self.demographic_dataset.subject_demographic.items():
            demographic_subject[v] = demographic_subject.get(v, []) + [k]

        # For each demographic, pic the negative pairs
        for d in demographic_subject:

            subject_combinations = itertools.combinations(
                demographic_subject[d], 2
            )

            for s_c in subject_combinations:
                subject_i = self.siamese_bucket[s_c[0]]
                subject_j = self.siamese_bucket[s_c[1]]
                random.shuffle(subject_i)
                random.shuffle(subject_j)

                # All possible combinations
                for i, p in enumerate(itertools.product(subject_i, subject_j)):
                    if i == self.negative_pairs_per_subject:
                        break
                    negatives += ((p[0], p[1]),)

        return negatives

    def create_light_negative_pairs(self):
        """
        Creating negative pairs.
        Here we create only negative pairs from the same demographic group,
        since we know that pairs from different demographics leads to
        poor scores

        .. warning:
            This function generates a light set of negative pairs.
            The number of pairs is composed by the number
            of subjects in a particular demographic
            multiplied by the number of `negative_pairs_per_subject`.
            For example, a combination pairs with 1000 identities gives us
            1000 pairs. Taking `3` pairs of images for these combinations
            of identities will give us 3000 negative pairs.

        """

        # Inverting subject
        random.seed(0)
        negatives = []

        # Creating the dictionary containing the demographics--> subjects
        demographic_subject = dict()
        for k, v in self.demographic_dataset.subject_demographic.items():
            demographic_subject[v] = demographic_subject.get(v, []) + [k]

        # For each demographic, pic the negative pairs

        for d in demographic_subject:

            n_subjects = len(demographic_subject[d])

            subject_combinations = list(
                itertools.combinations(demographic_subject[d], 2)
            )
            # Shuffling these combinations
            random.shuffle(subject_combinations)

            for s_c in subject_combinations[
                0 : n_subjects * self.negative_pairs_per_subject
            ]:
                subject_i = self.siamese_bucket[s_c[0]]
                subject_j = self.siamese_bucket[s_c[1]]
                random.shuffle(subject_i)
                random.shuffle(subject_j)

                negatives += ((subject_i[0], subject_j[0]),)

        return negatives

    def __getitem__(self, idx):

        sample = self.siamese_bucket[idx]
        label = self.labels[idx]

        # subject_id = sample.split("/")[-2]

        # Transforming the image
        image_i = sample[0].data
        image_j = sample[1].data

        image_i = (
            image_i
            if self.demographic_dataset.transform is None
            else self.demographic_dataset.transform(image_i)
        )
        image_j = (
            image_j
            if self.demographic_dataset.transform is None
            else self.demographic_dataset.transform(image_j)
        )

        demography = self.demographic_dataset.get_demographics(sample[0])

        # Getting the demographics

        # demography = self.get_demographics(subject_id)

        return {
            "data": (image_i, image_j),
            "label": label,
            "demography": demography,
        }
