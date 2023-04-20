#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  FRGC database implementation
"""

from clapper.rc import UserDefaults
from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDatabase, FileSampleLoader
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.pipelines import hash_string

rc = UserDefaults("bobrc.toml")


class FRGCDatabase(CSVDatabase):
    """
    Face Recognition Grand Test dataset
    """

    name = "frgc"
    category = "face"
    dataset_protocols_name = "frgc.tar.gz"
    dataset_protocols_urls = [
        "https://www.idiap.ch/software/bob/databases/latest/face/frgc-294a2ce4.tar.gz",
        "http://www.idiap.ch/software/bob/databases/latest/face/frgc-294a2ce4.tar.gz",
    ]
    dataset_protocols_hash = "294a2ce4"

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
                        "bob.bio.face.frgc.directory",  # TODO normalize this name
                        "",
                    ),
                    extension=rc.get("bob.bio.face.frgc.extension", ""),
                ),
                EyesAnnotations(),
            ),
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
            score_all_vs_all=True,
            memory_demanding=True,
        )

        self.hash_fn = hash_string
