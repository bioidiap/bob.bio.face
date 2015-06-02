#!/usr/bin/env python

import bob.db.arface
import bob.bio.base

arface_directory = "[YOUR_ARFACE_DIRECTORY]"

database = bob.bio.base.database.DatabaseBob(
    database = bob.db.arface.Database(
        original_directory = arface_directory
    ),
    name = 'arface',
    protocol = 'all'
)
