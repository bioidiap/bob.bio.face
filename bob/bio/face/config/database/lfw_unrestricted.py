#!/usr/bin/env python

import bob.db.lfw
import bob.bio.base

lfw_directory = "[YOUR_LFW_FUNNELED_DIRECTORY]"

database = bob.bio.base.database.DatabaseBob(
    database = bob.db.lfw.Database(
        original_directory = lfw_directory,
        annotation_type = 'funneled'
    ),
    name = 'lfw',
    protocol = 'view1',
    training_depends_on_protocol = True,
    models_depend_on_protocol = True,

    all_files_options = { 'world_type' : 'unrestricted' },
    extractor_training_options = { 'world_type' : 'unrestricted' }, # 'subworld' : 'twofolds'
    projector_training_options = { 'world_type' : 'unrestricted' }, # 'subworld' : 'twofolds'
    enroller_training_options =  { 'world_type' : 'unrestricted' } # 'subworld' : 'twofolds'
)
