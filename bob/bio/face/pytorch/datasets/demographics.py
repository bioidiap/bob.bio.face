#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


from torch.utils.data import Dataset

from bob.bio.face.database import MEDSDatabase, MorphDatabase, RFWDatabase


import torchvision.transforms as transforms


class DemoraphicTorchDataset(Dataset):
    def __init__(self, bob_dataset, transform=None):

        self.bob_dataset = bob_dataset
        self.bucket = [s for sset in self.bob_dataset.zprobes() for s in sset]
        self.bucket += [s for sset in self.bob_dataset.treferences() for s in sset]

        # Defining keys and labels
        keys = [sset.subject_id for sset in self.bob_dataset.zprobes()] + [
            sset.subject_id for sset in self.bob_dataset.treferences()
        ]
        self.labels = dict(zip(keys, range(len(keys))))

        self.demographic_keys = self.load_demographics()
        self.transform = transform

    def __len__(self):
        return len(self.bucket)

    def __getitem__(self, idx):

        sample = self.bucket[idx]

        image = sample.data if self.transform is None else self.transform(sample.data)

        # image = image.astype("float32")

        label = self.labels[sample.subject_id]

        demography = self.get_demographics(sample)

        return {"data": image, "label": label, "demography": demography}


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

    def load_demographics(self):

        target_metadata = "rac"
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
        demographic_key = getattr(sample, "rac")
        return self.demographic_keys[demographic_key]


class MorphTorchDataset(DemoraphicTorchDataset):
    def __init__(
        self, protocol, database_path, database_extension=".h5", transform=None
    ):

        self.bob_dataset = MorphDatabase(
            protocol=protocol,
            dataset_original_directory=database_path,
            dataset_original_extension=database_extension,
        )

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

        # Defining keys and labels
        keys = [b.subject_id for b in self.bucket]

        self.labels = dict(zip(keys, range(len(keys))))

        self.demographic_keys = self.load_demographics()
        self.transform = transform

        # super().__init__(bob_dataset, transform=transform)

    def load_demographics(self):

        target_metadata = "rac"

        metadata_keys = set(
            [f"{sset.rac}-{sset.sex}" for sset in self.bob_dataset.zprobes()]
            + [
                f"{sset.rac}-{sset.sex}"
                for sset in self.bob_dataset.treferences()
                if sset.subject_id not in self.excluding_list
            ]
        )
        metadata_keys = dict(zip(metadata_keys, range(len(metadata_keys))))
        return metadata_keys

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
