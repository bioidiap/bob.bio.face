from bob.bio.face.pytorch.datasets.webface42m import WebFace42M
from bob.bio.face.pytorch.datasets.demographics import (
    MedsTorchDataset,
    MorphTorchDataset,
    RFWTorchDataset,
    MobioTorchDataset,
    MSCelebTorchDataset,
)

from bob.extension import rc

# https://pytorch.org/docs/stable/data.html
from torch.utils.data import DataLoader
import os
import numpy as np

import pytest


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
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    batch = next(iter(dataloader))
    batch["data"].shape == (64, 3, 112, 112)

    weights = dataset.get_demographic_class_weights()

    assert np.allclose(sum(weights), 1)


@pytest.mark.skipif(
    rc.get("bob.bio.demographics.directory") is None,
    reason="Demographics features directory not available. Please do `bob config set bob.bio.demographics.directory [PATH]` to set the base features path.",
)
def test_rfw():

    database_path = os.path.join(
        rc.get("bob.bio.demographics.directory"), "rfw", "samplewrapper"
    )

    # RFW still not working

    # dataset = RFWTorchDataset(protocol="original", database_path=database_path,)

    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    batch = next(iter(dataloader))
    batch["data"].shape == (64, 3, 112, 112)


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


# @pytest.mark.skipif(
# rc.get("bob.bio.face.msceleb.directory") is None,
# reason="Demographics features directory not available. Please do `bob config set bob.bio.demographics.directory [PATH]` to set the base features path.",
# )
def test_msceleb():

    # database_path = os.path.join(
    #    rc.get("bob.bio.demographics.directory"), "mobio", "samplewrapper"
    # )

    ### WITH UNKNOW DEMOGRAPHICS
    database_path = "/idiap/temp/tpereira/databases/msceleb/112x112-eyes-crop/"

    dataset = MSCelebTorchDataset(database_path, include_unknow_demographics=True)
    assert dataset.n_classes == 89735
    assert len(dataset.demographic_keys) == 18

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    batch = next(iter(dataloader))
    batch["data"].shape == (64, 3, 112, 112)

    # WITHOUT UNKNOW DEMOGRAPHICS

    dataset = MSCelebTorchDataset(database_path, include_unknow_demographics=False)
    assert dataset.n_classes == 81279
    assert len(dataset.demographic_keys) == 15

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    batch = next(iter(dataloader))
    batch["data"].shape == (64, 3, 112, 112)

    weights = dataset.get_demographic_class_weights()

    assert np.allclose(sum(weights), 1)

    pass
