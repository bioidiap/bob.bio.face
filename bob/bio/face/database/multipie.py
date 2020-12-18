#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  Multipie database implementation 
"""

from bob.bio.base.database import CSVDataset
from bob.pipelines.datasets import CSVToSampleLoader
from bob.bio.face.database.sample_loaders import MultiposeAnnotations
from bob.extension import rc
from bob.extension.download import get_file
import bob.io.base
from sklearn.pipeline import make_pipeline


class MultipieDatabase(CSVDataset):
    """
    The Multipie database..
    """

    def __init__(self, protocol):

        # Downloading model if not exists
        urls = [
            "https://www.idiap.ch/software/bob/databases/latest/multipie.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/multipie.tar.gz",
        ]
        filename = get_file("multipie.tar.gz", urls)

        self.annotation_type = ["eyes-center", "left-profile", "right-profile"]
        self.fixed_positions = None

        super().__init__(
            filename,
            protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc["bob.db.multipie.directory"]
                    if rc["bob.db.multipie.directory"]
                    else "",
                    extension=".png",
                ),
                MultiposeAnnotations(),
            ),
        )

    @staticmethod
    def protocols():
        # TODO: Until we have (if we have) a function that dumps the protocols, let's use this one.
        return [
            "P240",
            "P191",
            "P130",
            "G",
            "P010",
            "P041",
            "P051",
            "P050",
            "M",
            "P110",
            "P",
            "P140",
            "U",
            "P200",
            "E",
            "P190",
            "P120",
            "P080",
            "P081",
            "P090",
        ]
