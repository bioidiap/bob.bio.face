#!/usr/bin/env python

import bob.db.caspeal
import bob.bio.base

caspeal_directory = "[YOUR_CAS-PEAL_DIRECTORY]"

database = bob.bio.base.database.DatabaseBob(
    database = bob.db.caspeal.Database(
        original_directory = caspeal_directory
    ),
    name = "caspeal",
    protocol = 'lighting'
)
