#!/usr/bin/env python


from bob.bio.face.database import MultipieBioDatabase

# here, we only want to have the cameras that are used in the P protocol
cameras = ('24_0', '01_0', '20_0', '19_0', '04_1', '05_0', '05_1', '14_0', '13_0', '08_0', '09_0', '12_0', '11_0')

multipie_image_directory = "[YOUR_MULTI-PIE_IMAGE_DIRECTORY]"
multipie_annotation_directory = "[YOUR_MULTI-PIE_ANNOTATION_DIRECTORY]"

database = MultipieBioDatabase(
    original_directory=multipie_image_directory,
    annotation_directory=multipie_annotation_directory,
    protocol = 'P',
    training_depends_on_protocol = True,

    all_files_options={'cameras': cameras},
    extractor_training_options={'cameras': cameras},
    projector_training_options={'cameras': cameras, 'world_sampling': 3, 'world_first': True},
    enroller_training_options={'cameras': cameras}

)

