#!/usr/bin/env python

import bob.db.scface
import bob.bio.base

scface_directory = "[YOUR_SC_FACE_DIRECTORY]"

# setup for SCface database
database = bob.bio.base.database.DatabaseBobZT(
    database = bob.db.scface.Database(
        original_directory = scface_directory
    ),
    name = 'scface',
    protocol = 'combined'
)
