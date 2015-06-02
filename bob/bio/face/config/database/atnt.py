#!/usr/bin/env python

import bob.db.atnt
import bob.bio.base

atnt_directory = "[YOUR_ATNT_DIRECTORY]"

database = bob.bio.base.database.DatabaseBob(
    database = bob.db.atnt.Database(
        original_directory = atnt_directory
    ),
    name = 'atnt'
)
