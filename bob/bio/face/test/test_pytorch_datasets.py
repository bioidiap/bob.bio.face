import os

import numpy as np
import pkg_resources
import pytest

# https://pytorch.org/docs/stable/data.html
from torch.utils.data import DataLoader

import bob.io.base

from bob.bio.face.pytorch.datasets import (
    MedsTorchDataset,
    MobioTorchDataset,
    MorphTorchDataset,
    MSCelebTorchDataset,
    SiameseDemographicWrapper,
    VGG2TorchDataset,
    WebFace42M,
)
from bob.bio.face.pytorch.preprocessing import get_standard_data_augmentation
from bob.extension import rc


@pytest.mark.skipif(
    rc.get("bob.bio.face.webface42M.directory") is None,
    reason="WEBFace42M  not available. Please do `bob config set bob.bio.face.ijbc.directory [IJBC PATH]` to set the IJBC data path.",
)
def test_webface42M():

    dataset = WebFace42M()

    sample = dataset[0]
    assert sample["label"] == 0
    assert sample["data"].shape == (3, 112, 112)

    sample = dataset[100000]
    assert sample["label"] == 4960
    assert sample["data"].shape == (3, 112, 112)

    sample = dataset[42474557]
    assert sample["label"] == 2059905
    assert sample["data"].shape == (3, 112, 112)

    sample = dataset[-1]
    assert sample["label"] == 2059905
    assert sample["data"].shape == (3, 112, 112)


@pytest.mark.skipif(
    rc.get("bob.bio.demographics.directory") is None,
    reason="Demographics features directory not available. Please do `bob config set bob.bio.demographics.directory [PATH]` to set the base features path.",
)
def test_meds():

    database_path = os.path.join(
        rc.get("bob.bio.demographics.directory"), "meds", "samplewrapper"
    )

    dataset = MedsTorchDataset(
        protocol="verification_fold1",
        database_path=database_path,
    )

    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=2
    )

    batch = next(iter(dataloader))
    batch["data"].shape == (64, 3, 112, 112)

    # Testing class weights
    weights = dataset.get_demographic_class_weights()
    assert np.allclose(sum(weights), 1)


@pytest.mark.skipif(
    rc.get("bob.bio.demographics.directory") is None,
    reason="Demographics features directory not available. Please do `bob config set bob.bio.demographics.directory [PATH]` to set the base features path.",
)
def test_morph():

    database_path = os.path.join(
        rc.get("bob.bio.demographics.directory"), "morph", "samplewrapper"
    )

    dataset = MorphTorchDataset(
        protocol="verification_fold1",
        database_path=database_path,
        take_from_znorm=False,
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    batch = next(iter(dataloader))
    batch["data"].shape == (64, 3, 112, 112)

    weights = dataset.get_demographic_class_weights()

    assert np.allclose(sum(weights), 1, atol=0.0001)


@pytest.mark.skipif(
    rc.get("bob.bio.demographics.directory") is None,
    reason="Demographics features directory not available. Please do `bob config set bob.bio.demographics.directory [PATH]` to set the base features path.",
)
def test_rfw():

    # database_path = os.path.join(
    #    rc.get("bob.bio.demographics.directory"), "rfw", "samplewrapper"
    # )

    # RFW still not working

    # dataset = RFWTorchDataset(protocol="original", database_path=database_path,)

    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # batch = next(iter(dataloader))
    # batch["data"].shape == (64, 3, 112, 112)
    pass


@pytest.mark.skipif(
    rc.get("bob.bio.demographics.directory") is None,
    reason="Demographics features directory not available. Please do `bob config set bob.bio.demographics.directory [PATH]` to set the base features path.",
)
def test_mobio():

    database_path = os.path.join(
        rc.get("bob.bio.demographics.directory"), "mobio", "samplewrapper"
    )

    dataset = MobioTorchDataset(
        protocol="mobile0-male-female",
        database_path=database_path,
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    batch = next(iter(dataloader))
    batch["data"].shape == (64, 3, 112, 112)

    # Testing class weights
    weights = dataset.get_demographic_class_weights()

    assert np.allclose(sum(weights), 1)


@pytest.mark.skipif(
    rc.get("bob.bio.face.msceleb.directory") is None,
    reason="Demographics features directory not available. Please do `bob config set bob.bio.face.msceleb.directory [PATH]` to set the base features path.",
)
def test_msceleb():

    database_path = rc.get("bob.bio.face.msceleb.directory")

    # WITH UNKNOW DEMOGRAPHICS
    dataset = MSCelebTorchDataset(
        database_path, include_unknow_demographics=True
    )
    assert dataset.n_classes == 89735
    assert len(dataset.demographic_keys) == 18

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    batch = next(iter(dataloader))
    batch["data"].shape == (64, 3, 112, 112)

    # WITHOUT UNKNOW DEMOGRAPHICS

    dataset = MSCelebTorchDataset(
        database_path, include_unknow_demographics=False
    )
    assert dataset.n_classes == 81279
    assert len(dataset.demographic_keys) == 15

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    batch = next(iter(dataloader))
    batch["data"].shape == (64, 3, 112, 112)

    weights = dataset.get_demographic_class_weights()

    assert np.allclose(sum(weights), 1, atol=0.001)


@pytest.mark.skipif(
    rc.get("bob.bio.face.vgg2-crops.directory") is None,
    reason="Demographics features directory not available. Please do `bob config set bob.bio.face.vgg2-crops.directory [PATH]` to set the base features path.",
)
def test_vgg2():

    database_path = rc.get("bob.bio.face.vgg2-crops.directory")

    dataset = VGG2TorchDataset(
        protocol="vgg2-short", database_path=database_path
    )

    assert np.allclose(
        sum(dataset.get_demographic_weights(as_dict=False)), 1, atol=1
    )

    assert dataset.n_classes == 8631
    assert len(dataset.demographic_keys) == 8

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    batch = next(iter(dataloader))
    batch["data"].shape == (64, 3, 112, 112)

    weights = dataset.get_demographic_class_weights()

    assert np.allclose(sum(weights), 1, atol=0.001)

    # Testing dev

    dataset = VGG2TorchDataset(
        protocol="vgg2-short", database_path=database_path, train=False
    )

    assert dataset.n_classes == 8631
    assert len(dataset.demographic_keys) == 8

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    batch = next(iter(dataloader))
    batch["data"].shape == (64, 3, 112, 112)

    weights = dataset.get_demographic_class_weights()

    assert np.allclose(sum(weights), 1, atol=0.001)


def test_data_augmentation():

    image = bob.io.base.load(
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/testimage.jpg"
        )
    )

    from bob.bio.face.pytorch.preprocessing import (
        get_standard_data_augmentation,
    )

    transform = get_standard_data_augmentation()
    transformed = transform(image)

    assert transformed.shape == (3, 531, 354)


@pytest.mark.skipif(
    rc.get("bob.bio.demographics.directory") is None,
    reason="Demographics features directory not available. Please do `bob config set bob.bio.demographics.directory [PATH]` to set the base features path.",
)
def test_siamese():
    siamese_transforms = get_standard_data_augmentation()

    # database_path = os.path.join(
    #    rc.get("bob.bio.demographics.directory"), "morph", "samplewrapper"
    # )

    database_path = rc.get("bob.bio.face.vgg2-crops.directory")

    # dataset = MobioTorchDataset(
    #    protocol="mobile0-male-female",
    #    database_path=database_path,
    #    transform=siamese_transforms,
    # )
    # dataset = MedsTorchDataset(
    #    protocol="verification_fold1",
    #    database_path=database_path,
    #    transform=siamese_transforms,
    #    take_from_znorm=False,
    # )
    dataset = VGG2TorchDataset(
        protocol="vgg2-short",
        database_path=database_path,
        database_extension=".jpg",
        transform=siamese_transforms,
    )

    # dataset = MorphTorchDataset(
    #    protocol="verification_fold1",
    #    database_path=database_path,
    #    transform=siamese_transforms,
    #    take_from_znorm=False,
    # )

    siamese_dataset = SiameseDemographicWrapper(
        dataset, max_positive_pairs_per_subject=5, negative_pairs_per_subject=3
    )

    dataloader = DataLoader(siamese_dataset, batch_size=64, shuffle=True)

    batch = next(iter(dataloader))

    batch["data"][0].shape == (64, 3, 112, 112)
    batch["data"][1].shape == (64, 3, 112, 112)
