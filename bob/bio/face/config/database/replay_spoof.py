#!/usr/bin/env python

from bob.bio.face.database import ReplayBioDatabase
from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector
from bob.extension import rc


replay_attack_directory = rc["bob.db.replay.directory"]

database = DatabaseConnector(
    ReplayBioDatabase(
        original_directory=replay_attack_directory,
        original_extension=".mov",
        protocol="grandtest-spoof",
    ),
    annotation_type="bounding-box",
    fixed_positions=None,
    # Only compare with spoofs from the same target identity
    allow_scoring_with_all_biometric_references=False,
)

