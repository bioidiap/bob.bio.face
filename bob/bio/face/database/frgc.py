#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  Multipie database implementation 
"""

from bob.bio.base.database import CSVDataset
from bob.bio.base.database import CSVToSampleLoaderBiometrics
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.extension import rc
from bob.extension.download import get_file
import bob.io.base
from sklearn.pipeline import make_pipeline


class FRGCDatabase(CSVDataset):
    """
    Face Recognition Grand Test dataset
    """

    def __init__(self, protocol, annotation_type="eyes-center", fixed_positions=None):

        # Downloading model if not exists
        urls = FRGCDatabase.urls()
        filename = get_file(
            "frgc.tar.gz", urls, file_hash="e675840204c80eec41ef49ccb1c39240304a5c4fc0f16306120e3418f0ba9fda"
        )

        super().__init__(
            name="frgc",
            dataset_protocol_path=filename,
            protocol=protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc.get("bob.db.frgc.directory", ""),
                    extension="",
                ),
                EyesAnnotations(),
            ),
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
        )

    @staticmethod
    def protocols():
        # TODO: Until we have (if we have) a function that dumps the protocols, let's use this one.
        return [
            "2.0.1",
            "2.0.2",
            "2.0.4",
        ]

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/frgc-e6758402.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/frgc-e6758402.tar.gz",
        ]
