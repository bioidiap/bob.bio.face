#!/usr/bin/env python

from bob.bio.face.database import IJBABioDatabase

ijba_directory = "[YOUR_IJBA_DIRECTORY]"

database = IJBABioDatabase(
  original_directory=ijba_directory,
  protocol='search_split1'
)
