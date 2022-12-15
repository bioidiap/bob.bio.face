#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent Colbois <laurent.colbois@idiap.ch>

"""
  SCFace database implementation
"""

from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.extension import rc
from bob.extension.download import get_file


class SCFaceDatabase(CSVDataset):
    """
    Surveillance Camera Face dataset

    SCface is a database of static images of human faces.\
    Images were taken in uncontrolled indoor environment using five video surveillance cameras of various qualities.
    Database contains 4160 static images (in visible and infrared spectrum) of 130 subjects.
    Images from different quality cameras mimic the real-world conditions and enable robust face recognition algorithms testing, emphasizing different
    law enforcement and surveillance use case scenarios.

    """

    def __init__(
        self, protocol, annotation_type="eyes-center", fixed_positions=None
    ):

        # Downloading model if not exists
        urls = SCFaceDatabase.urls()
        filename = get_file(
            "scface.tar.gz",
            urls,
            file_hash="813cd9339e3314826821978a11bdc34a",
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
            score_all_vs_all=True,
        )

    @staticmethod
    def protocols():
        # TODO: Until we have (if we have) a function that dumps the protocols, let's use this one.
        return ["close", "medium", "far", "combined", "IR"]

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/scface.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/scface.tar.gz",
        ]
