#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2020

from bob.pipelines.utils import hash_string
from bob.extension.download import get_file, find_element_in_tarball
import pickle
import os


def load_ijbc_sample(original_path, extension=[".jpg", ".png"]):
    for e in extension:
        path = original_path + e
        if os.path.exists(path):
            return path
    else:
        return ""


class IJBCDatabase:
    def __init__(self, pkl_directory=None):
        self.annotation_type = "bounding-box"
        self.fixed_positions = None
        self.allow_scoring_with_all_biometric_references = False
        self.hash_fn = hash_string
        self.memory_demanding = True

        if pkl_directory is None:
            urls = IJBCDatabase.urls()
            pkl_directory = get_file(
                "ijbc.tar.gz", urls, file_hash="c6061dc9fa021233366b3dd7fa205eb0"
            )

        self.pkl_directory = pkl_directory

    def _assert_group(self, group):
        assert (
            group == "dev"
        ), "The IJBC database only has a `dev` group. Received : {}".format(group)

    def references(self, group="dev"):
        self._assert_group(group)
        return pickle.loads(
            find_element_in_tarball(self.pkl_directory, "db_references.pickle", True)
        )

    def probes(self, group="dev"):
        self._assert_group(group)
        return pickle.loads(
            find_element_in_tarball(self.pkl_directory, "db_probes.pickle", True)
        )

    def background_model_samples(self):
        import cloudpickle

        return cloudpickle.loads(
            find_element_in_tarball(
                self.pkl_directory, "db_background_model_samples.pickle", True
            )
        )

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/ijbc.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/ijbc.tar.gz",
        ]
