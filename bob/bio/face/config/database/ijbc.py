#!/usr/bin/env python

from bob.bio.face.database import IJBCBioDatabase
from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector
from bob.extension import rc
import pickle
import os

ijbc_directory = rc["bob.db.ijbc.directory"]
ijbc_pkl_directory = rc["bob.db.ijbc.pkl_directory"]


class FastIJBCDatabase:
    def __init__(self, pkl_directory):
        self.pkl_directory = pkl_directory
        self.annotation_type='bounding-box'
        self.fixed_positions=None

    def _assert_group(self, group):
        assert group=="dev", "The IJBC database only has a `dev` group. Received : {}".format(group)

    def references(self, group="dev"):
        self._assert_group(group)
        return pickle.loads(open(os.path.join(self.pkl_directory, "db_references.pickle"), "rb").read())

    def probes(self, group="dev"):
        self._assert_group(group)
        return pickle.loads(open(os.path.join(self.pkl_directory, "db_probes.pickle"), "rb").read())

    def background_model_samples(self, group="dev"):
        self._assert_group(group)
        return pickle.loads(open(os.path.join(self.pkl_directory, "db_background_model_samples.pickle"), "rb").read())

database = FastIJBCDatabase(
    pkl_directory=ijbc_pkl_directory
)

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
