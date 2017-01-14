#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2016

"""
  CUHK_CUFS database implementation of bob.bio.base.database.ZTDatabase interface.
  It is an extension of an SQL-based database interface, which directly talks to CUHK_CUFS database, for
  verification experiments (good to use in bob.bio.base framework).
"""

from .database import FaceBioFile
from bob.bio.base.database import ZTBioDatabase


class CUHK_CUFSBioFile(FaceBioFile):

    def __init__(self, f):
        super(CUHK_CUFSBioFile, self).__init__(client_id=f.client_id, path=f.path, file_id=f.id)
        self._f = f


class CUHK_CUFSBioDatabase(ZTBioDatabase):
    """
    CUHK_CUFS database implementation of :py:class:`bob.bio.base.database.BioDatabase` interface.
    It is an extension of an SQL-based database interface, which directly talks to CUHK_CUFS database, for
    verification experiments (good to use in bob.bio.base framework).
    """

    def __init__(
            self,
            original_directory=None,
            original_extension=None,
            arface_directory="",
            xm2vts_directory="",
            **kwargs
    ):
        # call base class constructors to open a session to the database
        super(CUHK_CUFSBioDatabase, self).__init__(
            name='cuhk_cufs',
            original_directory=original_directory,
            original_extension=original_extension,
            arface_directory=arface_directory,
            xm2vts_directory=xm2vts_directory,
            **kwargs)

        from bob.db.cuhk_cufs.query import Database as LowLevelDatabase
        self._db = LowLevelDatabase(original_directory, original_extension, arface_directory, xm2vts_directory)

    def model_ids_with_protocol(self, groups=None, protocol="search_split1_p2s", **kwargs):
        return self._db.model_ids(groups=groups, protocol=protocol)

    def tmodel_ids_with_protocol(self, protocol="search_split1_p2s", groups=None, **kwargs):
        return self._db.tmodel_ids(protocol=protocol, groups=groups, **kwargs)

    def objects(self, groups=None, protocol="search_split1_p2s", purposes=None, model_ids=None, **kwargs):
        retval = self._db.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs)
        return [CUHK_CUFSBioFile(f) for f in retval]

    def tobjects(self, groups=None, protocol="search_split1_p2s", model_ids=None, **kwargs):
        retval = self._db.tobjects(groups=groups, protocol=protocol, model_ids=model_ids, **kwargs)
        return [CUHK_CUFSBioFile(f) for f in retval]

    def zobjects(self, groups=None, protocol="search_split1_p2s", **kwargs):
        retval = self._db.zobjects(groups=groups, protocol=protocol, **kwargs)
        return [CUHK_CUFSBioFile(f) for f in retval]

    def annotations(self, myfile):
        return self._db.annotations(myfile._f)
