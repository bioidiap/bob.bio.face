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

# from bob.bio.face.database.sample_loaders import eyes_annotations_loader
import os


cache_subdir = "datasets"
filename = "meds.tar.gz"
dataset_protocol_path = os.path.join(
    os.path.expanduser("~"), "bob_data", cache_subdir, filename
)


class MEDSDatabase(CSVDatasetZTNorm):
    """
    The MEDS-II (Multiple Encounter Data Set II) database interface

    .. warning::
        Use the command below to set the path of the real data::

            $ bob config set bob.db.meds.directory [PATH-TO-MEDS-DATA]

    Parameters
    ----------

    protocol: str
        One of the database protocols. Options are `verification_fold1`, `verification_fold2` and `verification_fold3`

    """

    def __init__(self, protocol):

        # Downloading model if not exists
        urls = [
            "https://www.idiap.ch/software/bob/databases/latest/meds.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/meds.tar.gz",
        ]
        get_file(filename, urls)

        self.annotation_type = ("eyes-center",)
        self.fixed_positions = None

        database = CSVDataset(
            dataset_protocol_path,
            protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc["bob.db.meds.directory"]
                    if rc["bob.db.meds.directory"]
                    else "",
                    extension=".jpg",
                ),
                EyesAnnotations(),
            ),
        )

        super().__init__(database)
