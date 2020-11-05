#!/usr/bin/env python

from bob.bio.face.database import MobioBioDatabase
from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector
from bob.extension import rc

database = DatabaseConnector(
    MobioBioDatabase(
        original_directory=rc["bob.db.mobio.directory"],
        annotation_directory=rc["bob.db.mobio.annotation_directory"],
        original_extension=".png",
        protocol="mobile0-male",
    )
)
database.allow_scoring_with_all_biometric_references = True


mobio_image_directory = rc["bob.db.mobio.directory"]
mobio_annotation_directory = rc["bob.db.mobio.annotation_directory"]

allow_scoring_with_all_biometric_references = True
annotation_type = "eyes-center"
fixed_positions = None


mobio_image = DatabaseConnector(
    MobioBioDatabase(
        original_directory=mobio_image_directory,
        original_extension=".png",
        annotation_directory=mobio_annotation_directory,
        annotation_type="eyecenter",
        protocol="male",
        models_depend_on_protocol=True,
    ),
    allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
    annotation_type=annotation_type,
    fixed_positions=fixed_positions,
)


mobio_male = DatabaseConnector(
    MobioBioDatabase(
        original_directory=mobio_image_directory,
        original_extension=".png",
        annotation_directory=mobio_annotation_directory,
        annotation_type="eyecenter",
        protocol="male",
        models_depend_on_protocol=True,
        all_files_options={"gender": "male"},
        extractor_training_options={"gender": "male"},
        projector_training_options={"gender": "male"},
        enroller_training_options={"gender": "male"},
        z_probe_options={"gender": "male"},
    ),
    allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
    annotation_type=annotation_type,
    fixed_positions=fixed_positions,
)


mobio_female = DatabaseConnector(
    MobioBioDatabase(
        original_directory=mobio_image_directory,
        original_extension=".png",
        annotation_directory=mobio_annotation_directory,
        annotation_type="eyecenter",
        protocol="female",
        models_depend_on_protocol=True,
        all_files_options={"gender": "female"},
        extractor_training_options={"gender": "female"},
        projector_training_options={"gender": "female"},
        enroller_training_options={"gender": "female"},
        z_probe_options={"gender": "female"},
    ),
    allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
    annotation_type=annotation_type,
    fixed_positions=fixed_positions,
)
