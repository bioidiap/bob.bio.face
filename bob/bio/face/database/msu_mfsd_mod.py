#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Amir Mohammadi <amir.mohammadi@idiap.ch>

from .database import FaceBioFile
from bob.bio.base.database import BioDatabase


class MsuMfsdModBioFile(FaceBioFile):
    """FaceBioFile implementation of the MSU_MFSD_MOD Database"""

    def __init__(self, f):
        super(MsuMfsdModBioFile, self).__init__(
            client_id=f.client_id, path=f.path, file_id=f.id)
        self._f = f

    def load(self, directory=None, extension=None):
        if extension in (None, '.mov', '.mp4'):
            return self._f.load(directory, extension)
        else:
            return super(MsuMfsdModBioFile, self).load(directory, extension)


class MsuMfsdModBioDatabase(BioDatabase):

    """
    MsuMfsdMod database implementation of
    :py:class:`bob.bio.base.database.BioDatabase` interface. It is an extension
    of an SQL-based database interface, which directly talks to MsuMfsdMod
    database, for verification experiments (good to use in bob.bio.base
    framework).
    """

    def __init__(self, max_number_of_frames=None, **kwargs):
        # call base class constructors to open a session to the database
        super(MsuMfsdModBioDatabase, self).__init__(
            name='msu-mfsd-mod',
            max_number_of_frames=max_number_of_frames, **kwargs)

        from bob.db.msu_mfsd_mod.verificationprotocol import Database \
            as LowLevelDatabase
        self._db = LowLevelDatabase(max_number_of_frames)

    def protocol_names(self):
        return self._db.protocols()

    def groups(self):
        return self._db.groups()

    def annotations(self, myfile):
        """
        Will return the bounding box annotation of nth frame of the video.
        """
        fn = myfile._f.framen
        # Frame index, 4 coordinates of the face rectangle (left, top, right,
        # bottom), 4 coordinates of the left and right eyes (xleft, yleft
        # xright, yright).
        annots = myfile._f._f.bbx(directory=self.original_directory)
        annotations = {'leye': (annots[fn][8], annots[fn][7]),
                       'reye': (annots[fn][6], annots[fn][5])}
        return annotations

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self._db.model_ids_with_protocol(groups, protocol, **kwargs)

    def objects(self, groups=None, protocol=None, purposes=None,
                model_ids=None, **kwargs):
        retval = self._db.objects(
            groups, protocol, purposes, model_ids, **kwargs)
        return [MsuMfsdModBioFile(f) for f in retval]

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
