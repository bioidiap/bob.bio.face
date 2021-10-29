#!/usr/bin/env python
# encoding: utf-8

import torchvision
from bob.extension import rc
import os

# https://pytorch.org/docs/stable/data.html
from torch.utils.data import DataLoader
from bob.bio.face.pytorch.lightning import BackboneHeadModel
from bob.learn.pytorch.architectures.lenet import Lenet5
from bob.bio.face.pytorch.head import ArcFace, Regular
from functools import partial
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

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


def test_boring_model():

    root_path = rc.get(
        "bob_data_folder", os.path.join(os.path.expanduser("~"), "bob_data")
    )

    train_dataloader = DataLoader(
        convert_dataset(
            torchvision.datasets.FashionMNIST(root_path, download=True, train=True)
        ),
        batch_size=128,
        shuffle=True,
        persistent_workers=True,
        num_workers=2,
    )
    validation_dataloader = DataLoader(
        convert_dataset(
            torchvision.datasets.FashionMNIST(root_path, download=True, train=False)
        ),
        batch_size=128,
    )

    backbone = Lenet5()
    # head = ArcFace(feat_dim=30, num_class=10)
    head = Regular(feat_dim=84, num_class=10)
    optimizer = partial(torch.optim.SGD, lr=0.1, momentum=0.9)

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
        amp_level="00",  # https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#amp-level
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
    )

    ## Assert the accuracy
    # assert trainer.validate()[0]["validation/accuracy"] > 0.5
