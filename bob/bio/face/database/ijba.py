#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2016

"""
  IJBA database implementation of bob.bio.base.database.Database interface.
  It is an extension of an SQL-based database interface, which directly talks to IJBA database, for
  verification experiments (good to use in bob.bio.base framework).
"""

from .database import FaceBioFile
from bob.bio.base.database import BioDatabase


class IJBABioFile(FaceBioFile):

    def __init__(self, f):
        super(IJBABioFile, self).__init__(client_id=f.client_id, path=f.path, file_id=f.id)
        self._f = f


class IJBABioDatabase(BioDatabase):
    """
    IJBA database implementation of :py:class:`bob.bio.base.database.BioDatabase` interface.
    It is an extension of an SQL-based database interface, which directly talks to IJBA database, for
    verification experiments (good to use in bob.bio.base framework).
    """

    def __init__(
            self,
            original_directory=None,
            annotations_directory=None,
            original_extension=None,
            **kwargs
    ):
        # call base class constructors to open a session to the database
        super(IJBABioDatabase, self).__init__(
            name='ijba',
            original_directory=original_directory,
            annotations_directory=annotations_directory,
            original_extension=original_extension,
            **kwargs)

        from bob.db.ijba.query import Database as LowLevelDatabase
        self._db = LowLevelDatabase(original_directory, annotations_directory,
                                    original_extension)

    def model_ids_with_protocol(self, groups=None, protocol="search_split1", **kwargs):
        return self._db.model_ids(groups=groups, protocol=protocol)

    def objects(self, groups=None, protocol="search_split1", purposes=None, model_ids=None, **kwargs):
        retval = self._db.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs)
        return [IJBABioFile(f) for f in retval]

    def annotations(self, myfile):
        return self._db.annotations(myfile._f)
