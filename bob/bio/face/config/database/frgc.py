#!/usr/bin/env python

import bob.db.frgc
import bob.bio.base

frgc_directory = "[YOUR_FRGC_DIRECTORY]"

database = bob.bio.base.database.DatabaseBob(
    database = bob.db.frgc.Database(frgc_directory),
    name = "frgc",
    protocol = '2.0.1',
    models_depend_on_protocol = True
)
