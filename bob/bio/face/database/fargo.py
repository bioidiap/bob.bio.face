#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Guillaume Heusch <guillaume.heusch@idiap.ch>
# Wed 04 Jan 15:08:22 CEST 2017

"""
  FARGO database implementation of bob.bio.base.database.Database interface.
  It is an extension of an SQL-based database interface, which directly talks to Atnt database, for
  verification experiments (good to use in bob.bio.base framework).
"""

from .database import FaceBioFile
from bob.bio.base.database import BioDatabase, BioFile


class FargoBioDatabase(BioDatabase):
    """
    Implements verification API for querying Atnt database.
    """

    def __init__(
            self,
            **kwargs
    ):
        # call base class constructors to open a session to the database
        super(FargoBioDatabase, self).__init__(name='fargo', **kwargs)

        from bob.db.fargo.query import Database as LowLevelDatabase
        self.__db = LowLevelDatabase()

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self.__db.model_ids(groups=groups, protocol=protocol)

    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        retval = self.__db.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs)
        return [FaceBioFile(client_id=f.client_id, path=f.path, file_id=f.id) for f in retval]
