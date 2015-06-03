#!/usr/bin/env python

import bob.db.xm2vts
import bob.bio.base

xm2vts_directory = "[YOUR_XM2VTS_DIRECTORY]"

# setup for XM2VTS
database = bob.bio.base.database.DatabaseBob(
    database = bob.db.xm2vts.Database(
        original_directory = xm2vts_directory
    ),
    name = "xm2vts",
    protocol = 'lp1'
)
