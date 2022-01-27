#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
Datasets that handles demographic information

"""


import logging
from torch.utils.data import Dataset

from bob.bio.face.database import (
    MEDSDatabase,
    MorphDatabase,
    RFWDatabase,
    MobioDatabase,
)


import torchvision.transforms as transforms

from bob.bio.demographics.script.demographics import demographics
import numpy as np
import pandas as pd
import os
import bob.io.base
import bob.io.image
from bob.extension.download import get_file
from bob.extension import rc

logger = logging.getLogger(__name__)


class DemoraphicTorchDataset(Dataset):
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
    def demographic_keys(self):
        return self._demographic_keys

    def __getitem__(self, idx):
        """
        It dumps a dictionary containing the following keys: data, label, demography

        """

        sample = self.bucket[idx]

        image = sample.data if self.transform is None else self.transform(sample.data)

        # image = image.astype("float32")

        label = self.labels[sample.subject_id]

        demography = self.get_demographics(sample)

        return {"data": image, "label": label, "demography": demography}

    def get_demographic_class_weights(self):
        """
        Compute the class weights based on the demographics

        Returns
        -------
            weights: list
              A list containing the weights for each class
        """

        n_identities = len(self.subject_demographic)

        all_demographics = list(self.subject_demographic.values())

        subjects_per_demographics = dict(
            [(d, sum(np.array(all_demographics) == d)) for d in set(all_demographics)]
        )

        # sum(np.array(all_demographics) == d)

        n_demographics = len(subjects_per_demographics)

        # I'll do it in 2 lines to make it readable
        weight_per_demographic = lambda x: (1 - (x / n_identities)) / (
            n_demographics - 1
        )

        weights_per_demographic = dict(
            [
                (d, weight_per_demographic(subjects_per_demographics[d]))
                for d in set(all_demographics)
            ]
        )

        weights = [
            weights_per_demographic[v] / subjects_per_demographics[v]
            for k, v in self.subject_demographic.items()
        ]

        return weights


class MedsTorchDataset(DemoraphicTorchDataset):
    def __init__(
        self, protocol, database_path, database_extension=".h5", transform=None
    ):

        bob_dataset = MEDSDatabase(
            protocol=protocol,
            dataset_original_directory=database_path,
            dataset_original_extension=database_extension,
        )
        super().__init__(bob_dataset, transform=transform)

    def load_bucket(self):
        self._target_metadata = "rac"

        self.bucket = [s for sset in self.bob_dataset.zprobes() for s in sset]
        self.bucket += [s for sset in self.bob_dataset.treferences() for s in sset]

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
        self._demographic_keys = dict(zip(metadata_keys, range(len(metadata_keys))))

    def get_demographics(self, sample):
        demographic_key = getattr(sample, "rac")
        return self._demographic_keys[demographic_key]


class MorphTorchDataset(DemoraphicTorchDataset):
    def __init__(
        self, protocol, database_path, database_extension=".h5", transform=None
    ):

        bob_dataset = MorphDatabase(
            protocol=protocol,
            dataset_original_directory=database_path,
            dataset_original_extension=database_extension,
        )
        super().__init__(bob_dataset, transform=transform)

    def load_bucket(self):

        # Morph dataset has an intersection in between zprobes and treferences
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

        self.bucket = [s for sset in self.bob_dataset.zprobes() for s in sset]
        self.bucket += [
            s
            for sset in self.bob_dataset.treferences()
            for s in sset
            if sset.subject_id not in self.excluding_list
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
        self._demographic_keys = dict(zip(metadata_keys, range(len(metadata_keys))))

    def get_demographics(self, sample):
        demographic_key = f"{sample.rac}-{sample.sex}"
        return self._demographic_keys[demographic_key]


class RFWTorchDataset(DemoraphicTorchDataset):
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
            [getattr(sset, target_metadata) for sset in self.bob_dataset.zprobes()]
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


class MobioTorchDataset(DemoraphicTorchDataset):
    def __init__(
        self, protocol, database_path, database_extension=".h5", transform=None
    ):

        bob_dataset = MobioDatabase(protocol=protocol)

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
        self._demographic_keys = dict(zip(metadata_keys, range(len(metadata_keys))))

    def __len__(self):
        return len(self.bucket)

    def get_demographics(self, sample):
        demographic_key = getattr(sample, self._target_metadata)
        return self._demographic_keys[demographic_key]


class MSCelebTorchDataset(DemoraphicTorchDataset):
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
                "bob_data_folder", os.path.join(os.path.expanduser("~"), "bob_data")
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

        filtered_dataframe_list = filtered_dataframe[["idiap_chunk", "ID"]].to_csv()

        # Defining the number of classes
        subject_relative_paths = [
            os.path.join(l.split(",")[1], l.split(",")[2])
            for l in filtered_dataframe_list.split("\n")[1:-1]
        ]

        if self.load_bucket_from_cache and os.path.exists(self.get_cache_path()):
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
            [(k.split("/")[-1], i) for i, k in enumerate(subject_relative_paths)]
        )

        ## Setting the possible demographics and the demographic keys
        filtered_dataframe = filtered_dataframe.set_index("ID")
        self.metadata = filtered_dataframe[["GENDER", "RACE"]].to_dict(orient="index")

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

        image = image.astype("float32")

        label = self.labels[subject_id]

        ## Getting the demographics

        demography = self.get_demographics(subject_id)

        return {"data": image, "label": label, "demography": demography}
