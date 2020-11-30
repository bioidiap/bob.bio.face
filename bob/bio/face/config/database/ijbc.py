#!/usr/bin/env python
import pickle
import os

ijbc_directory = "/idiap/resource/database/IJB-C/IJB/IJB-C/images"
ijbc_pkl_directory = "/idiap/temp/tpereira/ijbc_metadata/"
from bob.pipelines.utils import hash_string


class MetadataLinker:
    def __init__(self, name, protocol):
        self.name = name
        self.protocol = protocol

class FastIJBCDatabase:
    def __init__(self, pkl_directory):
        self.pkl_directory = pkl_directory
        self.annotation_type = "bounding-box"
        self.fixed_positions = None
        self.allow_scoring_with_all_biometric_references = False
        self.hash_fn = hash_string
        self.memory_demanding = True

    def _assert_group(self, group):
        assert (
            group == "dev"
        ), "The IJBC database only has a `dev` group. Received : {}".format(group)

    def references(self, group="dev"):
        self._assert_group(group)
        return pickle.loads(
            open(os.path.join(self.pkl_directory, "db_references.pickle"), "rb").read()
        )

    def probes(self, group="dev"):
        self._assert_group(group)
        return pickle.loads(
            open(os.path.join(self.pkl_directory, "db_probes.pickle"), "rb").read()
        )

    def background_model_samples(self, group="dev"):
        self._assert_group(group)
        return pickle.loads(
            open(
                os.path.join(self.pkl_directory, "db_background_model_samples.pickle"),
                "rb",
            ).read()
        )

database = FastIJBCDatabase(pkl_directory=ijbc_pkl_directory)

# database = DatabaseConnector(
#     IJBCBioDatabase(original_directory=ijbc_directory, protocol="1:1"),
#     annotation_type = "eyes-center",
#     fixed_positions = None,
# )

#ijbc_covariates = DatabaseConnector(
#    IJBCBioDatabase(
#        original_directory=ijbc_directory, protocol="Covariates"
#    )
#)
