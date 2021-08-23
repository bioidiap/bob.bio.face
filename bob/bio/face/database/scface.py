#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent Colbois <laurent.colbois@idiap.ch>

"""
  SCFace database implementation 
"""

from bob.bio.base.database import CSVDataset
from bob.bio.base.database import CSVToSampleLoaderBiometrics
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.extension import rc
from bob.extension.download import get_file
import bob.io.base
from sklearn.pipeline import make_pipeline


class SCFaceDatabase(CSVDataset):
    """
    Surveillance Camera Face dataset
    """

    def __init__(self, protocol, annotation_type="eyes-center", fixed_positions=None):

        # Downloading model if not exists
        urls = SCFaceDatabase.urls()
        filename = get_file(
            "scface.tar.gz",
            urls,
            file_hash="7f7b4f79fac4a734d79183c8ad568b957ff302f123ccf3d6ddc62384b7ba6ac4",
        )

        super().__init__(
            name="scface",
            dataset_protocol_path=filename,
            protocol=protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc.get(
                        "bob.bio.face.scface.directory", ""
                    ),
                    extension="",
                ),
                EyesAnnotations(),
            ),
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
            allow_scoring_with_all_biometric_references=True,
        )

    @staticmethod
    def protocols():
        # TODO: Until we have (if we have) a function that dumps the protocols, let's use this one.
        return ["close", "medium", "far", "combined", "IR"]

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/scface-7f7b4f79.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/scface-7f7b4f79.tar.gz",
        ]
