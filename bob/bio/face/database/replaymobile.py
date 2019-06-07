#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""   The Replay-Mobile Database for face spoofing implementation of
bob.bio.base.database.BioDatabase interface."""

from .database import FaceBioFile
from bob.bio.base.database import BioDatabase
from bob.extension import rc


class ReplayMobileBioFile(FaceBioFile):
    """FaceBioFile implementation of the Replay Mobile Database"""

    def __init__(self, f):
        super(ReplayMobileBioFile, self).__init__(client_id=f.client_id, path=f.path, file_id=f.id)
        self._f = f

    def load(self, directory=None, extension=None):
        if extension in (None, '.mov'):
            return self._f.load(directory, extension)
        else:
            return super(ReplayMobileBioFile, self).load(directory, extension)

    @property
    def annotations(self):
        return self._f.annotations


class ReplayMobileBioDatabase(BioDatabase):
    """
    ReplayMobile database implementation of :py:class:`bob.bio.base.database.BioDatabase` interface.
    It is an extension of an SQL-based database interface, which directly talks to ReplayMobile database, for
    verification experiments (good to use in bob.bio.base framework).
    """

    def __init__(self, max_number_of_frames=None,
                 annotation_directory=None,
                 annotation_extension='.json',
                 annotation_type='json',
                 original_directory=rc['bob.db.replaymobile.directory'],
                 original_extension='.mov',
                 name='replay-mobile',
                 **kwargs):
        from bob.db.replaymobile.verificationprotocol import Database as LowLevelDatabase
        self._db = LowLevelDatabase(
            max_number_of_frames,
            original_directory=original_directory,
            original_extension=original_extension,
            annotation_directory=annotation_directory,
            annotation_extension=annotation_extension,
            annotation_type=annotation_type,
        )

        # call base class constructors to open a session to the database
        super(ReplayMobileBioDatabase, self).__init__(
            name=name,
            original_directory=original_directory,
            original_extension=original_extension,
            annotation_directory=annotation_directory,
            annotation_extension=annotation_extension,
            annotation_type=annotation_type,
            **kwargs)
        self._kwargs['max_number_of_frames'] = max_number_of_frames

    @property
    def original_directory(self):
        return self._db.original_directory

    @original_directory.setter
    def original_directory(self, value):
        self._db.original_directory = value

    @property
    def original_extension(self):
        return self._db.original_extension

    @original_extension.setter
    def original_extension(self, value):
        self._db.original_extension = value

    @property
    def annotation_directory(self):
        return self._db.annotation_directory

    @annotation_directory.setter
    def annotation_directory(self, value):
        self._db.annotation_directory = value

    @property
    def annotation_extension(self):
        return self._db.annotation_extension

    @annotation_extension.setter
    def annotation_extension(self, value):
        self._db.annotation_extension = value

    @property
    def annotation_type(self):
        return self._db.annotation_type

    @annotation_type.setter
    def annotation_type(self, value):
        self._db.annotation_type = value

    def protocol_names(self):
        return self._db.protocols()

    def groups(self):
        return self._db.groups()

    def annotations(self, myfile):
        """Will return the bounding box annotation of nth frame of the video."""
        return myfile.annotations

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self._db.model_ids_with_protocol(groups, protocol, **kwargs)

    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        return [ReplayMobileBioFile(f) for f in self._db.objects(groups, protocol, purposes, model_ids, **kwargs)]

    def arrange_by_client(self, files):
        client_files = {}
        for file in files:
            if str(file.client_id) not in client_files:
                client_files[str(file.client_id)] = []
            client_files[str(file.client_id)].append(file)

        files_by_clients = []
        for client in sorted(client_files.keys()):
            files_by_clients.append(client_files[client])
        return files_by_clients
