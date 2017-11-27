#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Amir Mohammadi <amir.mohammadi@idiap.ch>
# Wed 13 Jul 16:43:22 CEST 2016

"""
  MOBIO database implementation of bob.bio.base.database.ZTBioDatabase interface.
  It is an extension of an SQL-based database interface, which directly talks to Mobio database, for
  verification experiments (good to use in bob.bio.base framework).
"""


from .database import FaceBioFile
from bob.bio.base.database import ZTBioDatabase


class MobioBioFile(FaceBioFile):
    """FaceBioFile implementation of the Mobio Database"""

    def __init__(self, f):
        super(MobioBioFile, self).__init__(client_id=f.client_id, path=f.path, file_id=f.id)
        self._f = f


class MobioBioDatabase(ZTBioDatabase):
    """
    MOBIO database implementation of bob.bio.base.database.ZTBioDatabase interface.
    It is an extension of an SQL-based database interface, which directly talks to Mobio database, for
    verification experiments (good to use in bob.bio.base framework).
    """

    def __init__(
            self,
            original_directory=None,
            original_extension=None,
            annotation_directory=None,
            annotation_extension='.pos',
            **kwargs
    ):
        from bob.db.mobio.query import Database as LowLevelDatabase
        self._db = LowLevelDatabase(original_directory, original_extension,
                                    annotation_directory, annotation_extension)

        # call base class constructors to open a session to the database
        super(MobioBioDatabase, self).__init__(
            name='mobio',
            original_directory=original_directory,
            original_extension=original_extension,
            annotation_directory=annotation_directory,
            annotation_extension=annotation_extension,
            **kwargs)

    @property
    def original_directory(self):
        return self._db.original_directory

    @original_directory.setter
    def original_directory(self, value):
        self._db.original_directory = value

    @property
    def annotation_directory(self):
        return self._db.annotation_directory

    @annotation_directory.setter
    def annotation_directory(self, value):
        self._db.annotation_directory = value

    def model_ids_with_protocol(self, groups=None, protocol=None, gender=None):
        return self._db.model_ids(groups=groups, protocol=protocol, gender=gender)

    def tmodel_ids_with_protocol(self, protocol=None, groups=None, **kwargs):
        return self._db.tmodel_ids(protocol=protocol, groups=groups, **kwargs)

    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        retval = self._db.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs)
        return [MobioBioFile(f) for f in retval]

    def tobjects(self, groups=None, protocol=None, model_ids=None, **kwargs):
        retval = self._db.tobjects(groups=groups, protocol=protocol, model_ids=model_ids, **kwargs)
        return [MobioBioFile(f) for f in retval]

    def zobjects(self, groups=None, protocol=None, **kwargs):
        retval = self._db.zobjects(groups=groups, protocol=protocol, **kwargs)
        return [MobioBioFile(f) for f in retval]

    def annotations(self, myfile):
        return self._db.annotations(myfile._f)
