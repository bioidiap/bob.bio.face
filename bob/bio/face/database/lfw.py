#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2016

"""
  LFW database implementation of bob.bio.base.database.Database interface.
  It is an extension of an SQL-based database interface, which directly talks to LFW database, for
  verification experiments (good to use in bob.bio.base framework).
"""


from .database import FaceBioFile
from bob.bio.base.database import BioDatabase


class LFWBioFile(FaceBioFile):

    def __init__(self, f):
        super(LFWBioFile, self).__init__(client_id=f.client_id, path=f.path, file_id=f.id)
        self._f = f


class LFWBioDatabase(BioDatabase):
    """
    LFW database implementation of bob.bio.base.database.Database interface.
    It is an extension of an SQL-based database interface, which directly talks to LFW database, for
    verification experiments (good to use in bob.bio.base framework).
    """

    def __init__(
            self,
            original_directory=None,
            original_extension='.jpg',
            annotation_type=None,
            **kwargs
    ):
        from bob.db.lfw.query import Database as LowLevelDatabase
        self._db = LowLevelDatabase(original_directory, original_extension, annotation_type)

        # call base class constructors to open a session to the database
        super(LFWBioDatabase, self).__init__(
            name='lfw',
            original_directory=original_directory,
            original_extension=original_extension,
            annotation_type=annotation_type,
            **kwargs)

    @property
    def original_directory(self):
        return self._db.original_directory

    @original_directory.setter
    def original_directory(self, value):
        self._db.original_directory = value

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self._db.model_ids(groups=groups, protocol=protocol)

    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        retval = self._db.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs)
        return [LFWBioFile(f) for f in retval]

    def annotations(self, myfile):
        return self._db.annotations(myfile._f)

    def client_id_from_model_id(self, model_id, group='dev'):
        """Return the client id associated with the given model id.
        In this base class implementation, it is assumed that only one model is enrolled for each client and, thus, client id and model id are identical.
        All key word arguments are ignored.
        Please override this function in derived class implementations to change this behavior."""

       # since there is one model per file, we can re-use the function above.
        return self._db.get_client_id_from_file_id(model_id)

