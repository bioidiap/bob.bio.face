#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  Multipie database implementation
"""

from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.extension import rc
from bob.extension.download import get_file
from bob.pipelines import hash_string


class FRGCDatabase(CSVDataset):
    """
    Face Recognition Grand Test dataset
    """

    def __init__(
        self, protocol, annotation_type="eyes-center", fixed_positions=None
    ):

        # Downloading model if not exists
        urls = FRGCDatabase.urls()
        filename = get_file(
            "frgc.tar.gz",
            urls,
            file_hash="242168e993fe0f6f29bd59fccf3c79a0",
        )

        super().__init__(
            name="frgc",
            dataset_protocol_path=filename,
            protocol=protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc.get(
                        "bob.bio.face.frgc.directory", ""
                    ),
                    extension="",
                    reference_id_equal_subject_id=False,
                ),
                EyesAnnotations(),
            ),
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
            score_all_vs_all=True,
            group_probes_by_reference_id=True,
            memory_demanding=True,
        )

        self.hash_fn = hash_string

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
            "https://www.idiap.ch/software/bob/databases/latest/frgc.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/frgc.tar.gz",
        ]
