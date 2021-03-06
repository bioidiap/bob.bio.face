#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  MEDS database implementation 
"""

from bob.bio.base.database import (
    CSVDataset,
    CSVDatasetZTNorm,
)
from bob.pipelines.datasets import CSVToSampleLoader
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.extension import rc
from bob.extension.download import get_file
import bob.io.base
from sklearn.pipeline import make_pipeline


class MorphDatabase(CSVDatasetZTNorm):
    """
    The MORPH dataset is relatively old, but is getting some traction recently mostly because its richness
    with respect to sensitive attributes.
    It is composed by 55,000 samples from 13,000 subjects from men and women and five
    race clusters (called ancestry) and they are the following: African, European, Asian, Hispanic and Others. Figure 8
    present some samples from this database.

    This dataset contains faces from five ethnicities (African, European, Asian, Hispanic, "Other")
    and two genders (Male and Female).
    Furthermore, this interface contains three verification protocols and they are:
    `verification_fold1`, `verification_fold2` and `verification_fold1`.
    Follow below the identities distribution in each set for the for each protocol:

    +--------------------+---------------+-----------+-----------+-----------+
    |                    | Training set              | Dev. Set  | Eval. Set |
    +--------------------+---------------+-----------+           +           +
    |                    | T-References  | Z-Probes  |           |           |
    +====================+===============+===========+===========+===========+
    | verification_fold1 |      69       |     66    |   6738    |   6742    |
    +--------------------+---------------+-----------+-----------+-----------+
    | verification_fold2 |      69       |     67    |   6734    |   6737    |
    +--------------------+---------------+-----------+-----------+-----------+
    | verification_fold3 |      70       |     66    |   6736    |   6740    |
    +--------------------+---------------+-----------+-----------+-----------+ 

    .. warning::
        Use the command below to set the path of the real data::

            $ bob config set bob.db.morph.directory [PATH-TO-MORPH-DATA]

    Parameters
    ----------

    protocol: str
        One of the database protocols. Options are `verification_fold1`, `verification_fold2` and `verification_fold3`

    """

    def __init__(self, protocol):

        # Downloading model if not exists
        urls = MorphDatabase.urls()
        filename = get_file("morph.tar.gz", urls)

        self.annotation_type = "eyes-center"
        self.fixed_positions = None

        database = CSVDataset(
            filename,
            protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc["bob.db.morph.directory"]
                    if rc["bob.db.morph.directory"]
                    else "",
                    extension=".JPG",
                ),
                EyesAnnotations(),
            ),
        )

        super().__init__(database)

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/morph.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/morph.tar.gz",
        ]
