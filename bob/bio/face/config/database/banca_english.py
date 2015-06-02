#!/usr/bin/env python

import bob.db.banca
import bob.bio.base

banca_directory = "[YOUR_BANCA_DIRECTORY]"

database = bob.bio.base.database.DatabaseBobZT(
    database = bob.db.banca.Database(
        original_directory = banca_directory,
        original_extension = '.ppm'
    ),
    name = "banca",
    protocol = 'P'
)
