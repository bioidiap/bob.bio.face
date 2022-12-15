#!/usr/bin/env python
# encoding: utf-8

import os

from functools import partial

import pytest
import torch
import torchvision

from torch import nn
from torch.nn import Module

# https://pytorch.org/docs/stable/data.html
from torch.utils.data import DataLoader, Dataset

from bob.bio.face.pytorch.head import ArcFace
from bob.bio.face.pytorch.lightning import BackboneHeadModel
from bob.extension import rc

# import torchvision.transforms as transforms


class MnistDictionaryDataset(Dataset):
    def __init__(self, fashion_mnist_dataset):
        super(MnistDictionaryDataset, self).__init__()

        self.fashion_mnist_dataset = fashion_mnist_dataset

    def __len__(self):
        return self.fashion_mnist_dataset.data.shape[0]

    def __getitem__(self, idx):
        return {
            "data": torch.unsqueeze(
                self.fashion_mnist_dataset.data[idx] / 255.0, axis=0
            ),
            "label": self.fashion_mnist_dataset.targets[idx],
        }


def convert_dataset(dataset):
    return MnistDictionaryDataset(dataset)


@pytest.mark.slow
def test_boring_model():
    import pytorch_lightning as pl

    class Lenet5Short(Module):
        def __init__(self, num_features=30):
            super(Lenet5Short, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(6, 8, 5)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(128, num_features)
            self.relu3 = nn.ReLU()

        def forward(self, x):
            y = self.conv1(x)
            y = self.relu1(y)
            y = self.pool1(y)
            y = self.conv2(y)
            y = self.relu2(y)
            y = self.pool2(y)
            y = y.view(y.shape[0], -1)
            y = self.fc1(y)
            y = self.relu3(y)
            return y

    root_path = rc.get(
        "bob_data_folder", os.path.join(os.path.expanduser("~"), "bob_data")
    )

    train_dataloader = DataLoader(
        convert_dataset(
            torchvision.datasets.FashionMNIST(
                root_path, download=True, train=True
            )
        ),
        batch_size=512,
        shuffle=True,
        persistent_workers=True,
        num_workers=2,
    )
    validation_dataloader = DataLoader(
        convert_dataset(
            torchvision.datasets.FashionMNIST(
                root_path, download=True, train=False
            )
        ),
        batch_size=128,
    )

    num_features = 30
    backbone = Lenet5Short(num_features=num_features)
    head = ArcFace(feat_dim=num_features, num_class=10)
    # head = Regular(feat_dim=84, num_class=10)
    optimizer = partial(torch.optim.SGD, lr=0.01, momentum=0.9)

    # Preparing lightining model
    model = BackboneHeadModel(
        backbone=backbone,
        head=head,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer_fn=optimizer,
    )

    # TODO: using this code to learn too
    # so, be nice with my comments
    trainer = pl.Trainer(
        # callbacks=..... # https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#callbacks
        # logger=logger,...
        max_epochs=1,
        gpus=-1 if torch.cuda.is_available() else None,
        # resume_from_checkpoint=resume_from_checkpoint, #https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#resume-from-checkpoint
        # debug flags
        # limit_train_batches=10,  # https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#limit-train-batches
        # limit_val_batches=1,
        # amp_level="00",  # https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#amp-level
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
    )

    acc = trainer.validate(dataloaders=validation_dataloader)[0][
        "validation/accuracy"
    ]

    assert acc > 0.2
