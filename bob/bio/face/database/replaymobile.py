#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""   The Replay-Mobile Database for face spoofing implementation of
bob.bio.base.database.BioDatabase interface."""

from .database import FaceBioFile
from bob.bio.base.database import BioDatabase


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


class ReplayMobileBioDatabase(BioDatabase):
    """
    ReplayMobile database implementation of :py:class:`bob.bio.base.database.BioDatabase` interface.
    It is an extension of an SQL-based database interface, which directly talks to ReplayMobile database, for
    verification experiments (good to use in bob.bio.base framework).
    """

    def __init__(self, max_number_of_frames=None, **kwargs):
        from bob.db.replaymobile.verificationprotocol import Database as LowLevelDatabase
        self._db = LowLevelDatabase(max_number_of_frames)

        # call base class constructors to open a session to the database
        super(ReplayMobileBioDatabase, self).__init__(
            name='replay-mobile', **kwargs)
        self._kwargs['max_number_of_frames'] = max_number_of_frames

    @property
    def original_directory(self):
        return self._db.original_directory

    @original_directory.setter
    def original_directory(self, value):
        self._db.original_directory = value

    def protocol_names(self):
        return self._db.protocols()

    def groups(self):
        return self._db.groups()

    def annotations(self, myfile):
        """Will return the bounding box annotation of nth frame of the video."""
        fn = myfile._f.framen
        annots = myfile._f._f.bbx(directory=self.original_directory)
        # convert width and height to bottomright coordinates
        # bob uses the (y, x) format
        topleft = (annots[fn][1], annots[fn][0])
        bottomright = (annots[fn][1] + annots[fn][3], annots[fn][0] + annots[fn][2])
        annotations = {'topleft': topleft, 'bottomright': bottomright}
        return annotations

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
