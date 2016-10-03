#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Pavel Korshunov <pavel.korshunov@idiap.ch>
# Mon 12 Oct 14:43:22 CEST 2015

"""
  Replay attack database implementation of bob.bio.base.database.BioDatabase interface.
  It is an extension of an SQL-based database interface, which directly talks to Replay database, for
  verification experiments (good to use in bob.bio.base framework).
  It also implements a kind of hack so that you can run vulnerability analysis with it.
"""

from .database import FaceBioFile
from bob.bio.base.database import BioDatabase


class ReplayBioFile(FaceBioFile):
    """docstring for ReplayBioFile"""
    def __init__(self, f):
        super(FaceBioFile, self).__init__(client_id=f.client_id, path=f.path, file_id=f.id)
        self._f = f

    def load(self, directory=None, extension=None):
        video = self._f.load(directory, extension)
        # just return the 10th frame.
        return video[10]


class ReplayBioDatabase(BioDatabase):
    """
    Implements verification API for querying Replay database.
    This database only loads the 10th image from the video files
    """
    __doc__ = __doc__

    def __init__(self, **kwargs):
        # call base class constructors to open a session to the database
        super(ReplayBioDatabase, self).__init__(name='replay', **kwargs)

        from bob.db.replay import Database as LowLevelDatabase
        self.__db = LowLevelDatabase()

        self.low_level_group_names = ('train', 'devel', 'test')
        self.high_level_group_names = ('world', 'dev', 'eval')

    def protocol_names(self):
        """Returns all registered protocol names
        Here I am going to hack and double the number of protocols
        with -licit and -spoof. This is done for running vulnerability
        analysis"""
        names = [p.name + '-licit' for p in self.__db.protocols()]
        names += [p.name + '-spoof' for p in self.__db.protocols()]
        return names

    def groups(self):
        return self.convert_names_to_highlevel(
            self.__db.groups(), self.low_level_group_names, self.high_level_group_names)

    def annotations(self, file):
        """Will return the bounding box annotation of 10th frame of the video."""
        fn = 10  # 10th frame number
        annots = file._f.bbx(directory=self.original_directory)
        # bob uses the (y, x) format
        topleft = (annots[fn][2], annots[fn][1])
        bottomright = (annots[fn][2] + annots[fn][4], annots[fn][1] + annots[fn][3])
        annotations = {'topleft': topleft, 'bottomright': bottomright}
        return annotations

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        # since the low-level API does not support verification straight-forward-ly, we improvise.
        files = self.objects(groups=groups, protocol=protocol)
        return sorted(set(f.client_id for f in files))

    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        if protocol == '.':
            protocol = None
        protocol = self.check_parameter_for_validity(protocol, "protocol", self.protocol_names(), 'grandtest-licit')
        groups = self.check_parameters_for_validity(groups, "group", self.groups(), self.groups())
        purposes = self.check_parameters_for_validity(purposes, "purpose", ('enroll', 'probe'), ('enroll', 'probe'))
        purposes = list(purposes)
        groups = self.convert_names_to_lowlevel(
            groups, self.low_level_group_names, self.high_level_group_names)

        # protocol licit is not defined in the low level API
        # so do a hack here.
        if '-licit' in protocol:
            # for licit we return the grandtest protocol
            protocol = protocol.replace('-licit', '')
            # The low-level API has only "attack", "real", "enroll" and "probe"
            # should translate to "real" or "attack" depending on the protocol.
            # enroll does not to change.
            if 'probe' in purposes:
                purposes.remove('probe')
                purposes.append('real')
                if len(purposes) == 1:
                    # making the model_ids to None will return all clients which make
                    # the impostor data also available.
                    model_ids = None
                elif model_ids:
                    raise NotImplementedError(
                       'Currently returning both enroll and probe for specific '
                       'client(s) in the licit protocol is not supported. '
                       'Please specify one purpose only.')
        elif '-spoof' in protocol:
            protocol = protocol.replace('-spoof', '')
            # you need to replace probe with attack and real for the spoof protocols.
            # I am adding the real here also to create positives scores also.
            if 'probe' in purposes:
                purposes.remove('probe')
                purposes.append('attack')

        # now, query the actual Replay database
        objects = self.__db.objects(groups=groups, protocol=protocol, cls=purposes, clients=model_ids, **kwargs)

        # make sure to return BioFile representation of a file, not the database one
        # also make sure you replace client ids with spoof/metatdata1/metadata2/...
        retval = []
        for f in objects:
            if f.is_real():
                retval.append(ReplayBioFile(f))
            else:
                temp = ReplayBioFile(f)
                temp.client_id = 'attack'
                retval.append(temp)
        return retval
