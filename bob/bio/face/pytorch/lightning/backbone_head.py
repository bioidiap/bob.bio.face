import os

import numpy as np
import pytorch_lightning as pl
import scipy.spatial
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau


class BackboneHeadModel(pl.LightningModule):
    def __init__(
        self,
        backbone,
        head,
        loss_fn,
        optimizer_fn,
        backbone_checkpoint_file=None,
        **kwargs
    ):
        """
        Pytorch-lightining (https://pytorch-lightning.readthedocs.io/) model composed of two `torch.nn.Module`:
        `backbone` and `head`.


        Use this model if you want to compose a lightning model that mixing a standard backbone
        (Resnet, InceptionResnet, EfficientNet....) and a head (ArcFace, regular cross entropy).


        .. note::
          The `validation_step` of this module runs a validation in the level of embeddings, doing
          closed-set identification.
          Hence, it's mandatory to have a validation dataloader containg pairs of samples of the same identity in a sequence



        Parameters
        ----------

            backbone: `torch.nn.Module`
              Backbone module

            head: `torch.nn.Module`
              Head module

            loss_fn:
                A loss function

            optimizer_fn:
                A `torch.optim` function

            backbone_checkpoint_path:
                Path for the backbone


        Example
        -------

        Follow below


        """

        super().__init__(**kwargs)
        self.backbone = backbone
        self.head = head
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.backbone_checkpoint_file = backbone_checkpoint_file

    def forward(self, inputs):
        # in lightning, forward defines the prediction/inference actions
        return self.backbone(inputs)

    def training_epoch_end(self, training_step_outputs):

        if self.backbone_checkpoint_file is not None:

            torch.save(
                self.backbone.state_dict(),
                os.path.join(self.backbone_checkpoint_file),
            )

    # def training_step_end(self, losses):
    #    pass

    def validation_step(self, val_batch, batch_idx):

        data = val_batch["data"]
        labels = val_batch["label"].cpu().detach().numpy()

        val_embedding = torch.nn.functional.normalize(self.forward(data), p=2)
        val_embedding = val_embedding.cpu().detach().numpy()
        n = val_embedding.shape[0]

        # Distance with all vectors in a batch
        pdist = scipy.spatial.distance.pdist(val_embedding, metric="cosine")

        # Squared matrix with infiity
        predictions = np.ones((n, n)) * np.inf

        # Filling the upper triangular (without the diagonal) with the pdist
        predictions[np.triu_indices(n, k=1)] = pdist

        # predicting
        predictions = labels[np.argmin(predictions, axis=1)]

        accuracy = sum(predictions == labels) / n
        self.log("validation/accuracy", accuracy)

    def training_step(self, batch, batch_idx):

        data = batch["data"]
        label = batch["label"]

        embedding = self.backbone(data)

        logits = self.head(embedding, label)

        loss = self.loss_fn(logits, label)

        self.log("train/loss", loss)

        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        config = dict()
        optimizer = self.optimizer_fn(params=self.parameters())
        config["optimizer"] = optimizer

        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
        config["lr_scheduler"] = lr_scheduler

        config["monitor"] = "train/loss"

        return config
