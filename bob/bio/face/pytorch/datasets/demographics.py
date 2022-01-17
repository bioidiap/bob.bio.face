#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


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


class DemoraphicTorchDataset(Dataset):
    def __init__(self, bob_dataset, transform=None):

        self.bob_dataset = bob_dataset
        self.transform = transform
        self.load_bucket()

    def __len__(self):
        return len(self.bucket)

    def __getitem__(self, idx):

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

        # weight_per_demographic = 1-(samples_per_demographic/n_classes) / (n_demographics-1)
        # weight per subject = weight_per_demographic/samples_per_demographic
        #

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
        self.demographic_keys = dict(zip(metadata_keys, range(len(metadata_keys))))

    def get_demographics(self, sample):
        demographic_key = getattr(sample, "rac")
        return self.demographic_keys[demographic_key]


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
        self.demographic_keys = dict(zip(metadata_keys, range(len(metadata_keys))))

    def get_demographics(self, sample):
        demographic_key = f"{sample.rac}-{sample.sex}"
        return self.demographic_keys[demographic_key]


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
        return self.demographic_keys[demographic_key]


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
        self.demographic_keys = dict(zip(metadata_keys, range(len(metadata_keys))))

    def __len__(self):
        return len(self.bucket)

    def __getitem__(self, idx):

        sample = self.bucket[idx]

        image = sample.data if self.transform is None else self.transform(sample.data)

        # image = image.astype("float32")

        label = self.labels[sample.subject_id]

        demography = self.get_demographics(sample)

        return {"data": image, "label": label, "demography": demography}

    def get_demographics(self, sample):
        demographic_key = getattr(sample, self._target_metadata)
        return self.demographic_keys[demographic_key]
