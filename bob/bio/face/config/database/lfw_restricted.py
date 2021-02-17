#!/usr/bin/env python

from bob.bio.face.database import LFWBioDatabase
from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector
from bob.extension import rc


lfw_directory = rc["bob.db.lfw.directory"]

database = DatabaseConnector(
    LFWBioDatabase(
        original_directory=lfw_directory,
        annotation_type="funneled",
        protocol="view1",
        training_depends_on_protocol=True,
        models_depend_on_protocol=True,
        all_files_options={"world_type": "restricted"},
        extractor_training_options={
            "world_type": "restricted"
        },  # 'subworld' : 'twofolds'
        projector_training_options={
            "world_type": "restricted"
        },  # 'subworld' : 'twofolds'
        enroller_training_options={
            "world_type": "restricted"
        },  # 'subworld' : 'twofolds'
    ),
    allow_scoring_with_all_biometric_references=False,
)
