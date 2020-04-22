#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2016

"""
  IJBC database implementation of bob.bio.base.database.BioDatabase interface.
  It is an extension of the database interface, which directly talks to IJBC database, for
  verification experiments (good to use in bob.bio.base framework).
"""

from .database import FaceBioFile
from bob.bio.base.database import BioDatabase, BioFileSet
import os
import bob.io.image


class IJBCBioFile(FaceBioFile):
    def __init__(self, f):
        super(IJBCBioFile, self).__init__(
            client_id=f.client_id, path=f.path, file_id=f.id)
        self.f = f

    def load(self, directory, extension=None, add_client_id=False):
        return bob.io.image.load(self.make_path(directory, self.f.extension, add_client_id))

    def make_path(self, directory, extension, add_client_id=True):
        if add_client_id:
            # add client ID to the path, so that a unique path is generated
            # (there might be several identities in each physical file)
            path = "%s-%s%s" % (self.path, self.client_id, extension or '')
        else:
            # do not add the client ID to be able to obtain the original image file
            path = "%s%s" % (self.path,  extension or '')

        return str(os.path.join(directory or '', path))


class IJBCBioFileSet(BioFileSet):
    def __init__(self, template):
        super(IJBCBioFileSet, self).__init__(file_set_id=template.id, files=[
            IJBCBioFile(f) for f in template.files], path=template.path)


class IJBCBioDatabase(BioDatabase):
    """
      IJBC database implementation of :py:class:`bob.bio.base.database.BioDatabase` interface.
      It is an extension of an SQL-based database interface, which directly talks to IJBC database, for
      verification experiments (good to use in bob.bio.base framework).
    """

    def __init__(
        self,
        original_directory=None,
        annotation_directory=None,
        original_extension=None,
        **kwargs
    ):
        from bob.db.ijbc.query import Database as LowLevelDatabase
        self._db = LowLevelDatabase(
            original_directory)

        # call base class constructors to open a session to the database
        super(IJBCBioDatabase, self).__init__(
            name='ijbc',
            models_depend_on_protocol=True,
            training_depends_on_protocol=True,
            original_directory=original_directory,
            annotation_directory=annotation_directory,
            original_extension=original_extension,
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

    def uses_probe_file_sets(self):
        return True

    def model_ids_with_protocol(self, groups=None, protocol="1:1", **kwargs):
        return self._db.model_ids(groups=groups, protocol=protocol)

    def objects(self, groups=None, protocol="1:1", purposes=None, model_ids=None, **kwargs):
        return [IJBCBioFile(f) for f in self._db.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs)]

    def object_sets(self, groups=None, protocol="1:1", purposes=None, model_ids=None):
        return [IJBCBioFileSet(t) for t in self._db.object_sets(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids)]

    def annotations(self, biofile):
        return self._db.annotations(biofile.f)

    def client_id_from_model_id(self, model_id, group='dev'):
        return self._db.get_client_id_from_model_id(self.protocol, model_id)

    def original_file_names(self, files):
        return [f.make_path(self.original_directory, f.f.extension, False) for f in files]
