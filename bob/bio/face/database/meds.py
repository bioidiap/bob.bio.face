#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  MEDS database implementation 
"""

from bob.bio.base.database import CSVDatasetDevEval, CSVToSampleLoader
from bob.extension import rc
import bob.io.base
from bob.bio.face.database.sample_loaders import EyesAnnotationsLoader


# TODO: POINT TO THE `.bob/meds``
dataset_protocol_path = "/idiap/user/tpereira/gitlab/bob/bob.nightlies/meds"


class MEDSDatabase(CSVDatasetDevEval):
    def __init__(
        self,
        protocol,
        dataset_protocol_path=dataset_protocol_path,
        csv_to_sample_loader=CSVToSampleLoader(
            data_loader=bob.io.base.load,
            metadata_loader=EyesAnnotationsLoader(),
            dataset_original_directory=rc["bob.db.meds.directory"],
            extension=".jpg",
        ),
    ):

        # TODO: IMPLEMENT THE DOWNLOAD MECHANISM

        super().__init__(dataset_protocol_path, protocol, csv_to_sample_loader)

