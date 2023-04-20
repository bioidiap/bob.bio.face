#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent Colbois <laurent.colbois@idiap.ch>

"""
  SCFace database implementation
"""

from clapper.rc import UserDefaults
from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDatabase, FileSampleLoader
from bob.bio.face.database.sample_loaders import EyesAnnotations

rc = UserDefaults("bobrc.toml")


class SCFaceDatabase(CSVDatabase):
    """
    Surveillance Camera Face dataset

    SCface is a database of static images of human faces.\
    Images were taken in uncontrolled indoor environment using five video surveillance cameras of various qualities.
    Database contains 4160 static images (in visible and infrared spectrum) of 130 subjects.
    Images from different quality cameras mimic the real-world conditions and enable robust face recognition algorithms testing, emphasizing different
    law enforcement and surveillance use case scenarios.

    """

    name = "scface"
    category = "face"
    dataset_protocols_name = "scface.tar.gz"
    dataset_protocols_urls = [
        "https://www.idiap.ch/software/bob/databases/latest/face/scface-e6ffa822.tar.gz",
        "http://www.idiap.ch/software/bob/databases/latest/face/scface-e6ffa822.tar.gz",
    ]
    dataset_protocols_hash = "e6ffa822"

    def __init__(
        self, protocol, annotation_type="eyes-center", fixed_positions=None
    ):
        super().__init__(
            name=self.name,
            protocol=protocol,
            transformer=make_pipeline(
                FileSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc.get(
                        "bob.bio.face.scface.directory", ""
                    ),
                    extension=rc.get("bob.bio.face.scface.extension", ""),
                ),
                EyesAnnotations(),
            ),
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
            score_all_vs_all=True,
        )
