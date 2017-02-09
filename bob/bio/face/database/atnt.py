#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Amir Mohammadi <amir.mohammadi@idiap.ch>
# Wed 13 Jul 16:43:22 CEST 2016

"""
  Atnt database implementation of bob.bio.base.database.Database interface.
  It is an extension of an SQL-based database interface, which directly talks to Atnt database, for
  verification experiments (good to use in bob.bio.base framework).
"""

from .database import FaceBioFile
from bob.bio.base.database import BioDatabase


class AtntBioDatabase(BioDatabase):
    """
    ATNT database implementation of :py:class:`bob.bio.base.database.BioDatabase` interface.
    It is an extension of the database interface, which directly talks to ATNT database, for
    verification experiments (good to use in bob.bio.base framework).
    """

    def __init__(
            self,
            original_directory=None,
            original_extension='.pgm',
            **kwargs
    ):
        # call base class constructors to open a session to the database
        super(AtntBioDatabase, self).__init__(
            name='atnt',
            original_directory=original_directory,
            original_extension=original_extension,
            **kwargs)

        from bob.db.atnt.query import Database as LowLevelDatabase
        self._db = LowLevelDatabase(original_directory, original_extension)

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self._db.model_ids(groups=groups, protocol=protocol)

    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        retval = self._db.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs)
        return [FaceBioFile(client_id=f.client_id, path=f.path, file_id=f.id) for f in retval]

    def annotations(self, file):
        return None
