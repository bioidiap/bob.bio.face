#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2016

import os
import xml.sax

from functools import partial

import bob.io.base

from bob.bio.base.pipelines.abstract_classes import Database
from bob.extension import rc
from bob.extension.download import get_file, search_file
from bob.pipelines import DelayedSample, SampleSet

"""
GBU Database

Several of the rules used in this code were imported from
https://gitlab.idiap.ch/bob/bob.db.gbu/-/blob/master/bob/db/gbu/create.py
"""


def load_annotations(annotations_file):
    annotations = dict()
    for i, line in enumerate(annotations_file.readlines()):
        # Skip the first line
        if i == 0:
            continue
        line = line.split(",")
        path = os.path.splitext(os.path.basename(line[0]))[0]
        annotations[path] = {
            "leye": (float(line[-1]), float(line[-2])),
            "reye": (float(line[2]), float(line[1])),
        }
    return annotations


class File(object):
    def __init__(self, subject_id, reference_id, path):
        self.subject_id = subject_id
        self.reference_id = reference_id
        self.path = path


class XmlFileReader(xml.sax.handler.ContentHandler):
    def __init__(self):
        self.m_signature = None
        self.m_path = None
        self.m_presentation = None
        self.m_file_list = dict()

    def startDocument(self):
        pass

    def endDocument(self):
        pass

    def startElement(self, name, attrs):
        if name == "biometric-signature":
            self.m_signature = attrs["name"]  # subject_id
        elif name == "presentation":
            self.m_path = os.path.splitext(attrs["file-name"])[0]  # path
            self.m_presentation = attrs["name"]  # reference_id
        else:
            pass

    def endElement(self, name):
        if name == "biometric-signature":
            # assert that everything was read correctly
            assert (
                self.m_signature is not None
                and self.m_path is not None
                and self.m_presentation is not None
            )
            # add a file to the sessions
            self.m_file_list[self.m_presentation] = File(
                subject_id_from_signature(self.m_signature),
                self.m_presentation,
                self.m_path,
            )

            self.m_presentation = self.m_signature = self.m_path = None
        else:
            pass


def subject_id_from_signature(signature):
    return int(signature[4:])


def read_list(xml_file, eye_file=None):
    """Reads the xml list and attaches the eye files, if given"""
    # create xml reading instance
    handler = XmlFileReader()
    xml.sax.parse(xml_file, handler)
    return handler.m_file_list


class GBUDatabase(Database):
    """
    The GBU (Good, Bad and Ugly) database consists of parts of the MBGC-V1 image set.
    It defines three protocols, i.e., `Good`, `Bad` and `Ugly` for which different model and probe images are used.


    .. warning::

      To use this dataset protocol, you need to have the original files of the IJBC datasets.
      Once you have it downloaded, please run the following command to set the path for Bob

        .. code-block:: sh

            bob config set bob.bio.face.gbu.directory [GBU PATH]


    The code below allows you to fetch the galery and probes of the "Good" protocol.

    .. code-block:: python

        >>> from bob.bio.face.database import GBUDatabase
        >>> gbu = GBUDatabase(protocol="Good")
        >>>
        >>> # Fetching the gallery
        >>> references = gbu.references()
        >>> # Fetching the probes
        >>> probes = gbu.probes()


    """

    def __init__(
        self,
        protocol,
        annotation_type="eyes-center",
        fixed_positions=None,
        original_directory=rc.get("bob.bio.face.gbu.directory"),
        extension=".jpg",
    ):

        # self.filename = "/idiap/user/tpereira/gitlab/bob/bob.nightlies/temp/gbu.tar.gz"
        # Downloading model if not exists
        urls = GBUDatabase.urls()
        self.filename = get_file(
            "gbu-xmls.tar.gz",
            urls,
            file_hash="827de43434ee84020c6a949ece5e4a4d",
        )

        self.references_dict = {}
        self.probes_dict = {}

        self.annotations = None
        self.original_directory = original_directory
        self.extension = extension

        self.background_samples = None
        self._background_files = [
            "GBU_Training_Uncontrolledx1.xml",
            "GBU_Training_Uncontrolledx2.xml",
            "GBU_Training_Uncontrolledx4.xml",
            "GBU_Training_Uncontrolledx8.xml",
        ]

        super().__init__(
            name="gbu",
            protocol=protocol,
            score_all_vs_all=True,
            annotation_type="eyes-center",
            fixed_positions=fixed_positions,
            memory_demanding=True,
        )

    @staticmethod
    def protocols():
        return ["Good", "Bad", "Ugly"]

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/gbu-xmls.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/gbu-xmls.tar.gz",
        ]

    def background_model_samples(self):
        if self.background_samples is None:
            if self.annotations is None:
                self.annotations = load_annotations(
                    search_file(self.filename, "alleyes.csv")
                )
            # for
            self.background_samples = []

            for b_files in self._background_files:

                f = search_file(self.filename, f"{b_files}")

                self.background_samples += self._make_sampleset_from_filedict(
                    read_list(f)
                )
        return self.background_samples

    def probes(self, group="dev"):
        if self.protocol not in self.probes_dict:
            if self.annotations is None:
                self.annotations = load_annotations(
                    search_file(self.filename, "alleyes.csv")
                )

            f = search_file(self.filename, f"GBU_{self.protocol}_Query.xml")
            reference_ids = [x.reference_id for x in self.references()]

            self.probes_dict[
                self.protocol
            ] = self._make_sampleset_from_filedict(read_list(f), reference_ids)
        return self.probes_dict[self.protocol]

    def references(self, group="dev"):

        if self.protocol not in self.references_dict:

            if self.annotations is None:
                self.annotations = load_annotations(
                    search_file(self.filename, "alleyes.csv")
                )

            f = search_file(self.filename, f"GBU_{self.protocol}_Target.xml")
            self.references_dict[
                self.protocol
            ] = self._make_sampleset_from_filedict(
                read_list(f),
            )

        return self.references_dict[self.protocol]

    def groups(self):
        return ["dev"]

    def all_samples(self, group="dev"):
        self._check_group(group)

        return self.references() + self.probes()

    def _check_protocol(self, protocol):
        assert (
            protocol in self.protocols()
        ), "Unvalid protocol `{}` not in {}".format(protocol, self.protocols())

    def _check_group(self, group):
        assert group in self.groups(), "Unvalid group `{}` not in {}".format(
            group, self.groups()
        )

    def _make_sampleset_from_filedict(self, file_dict, reference_ids=None):
        samplesets = []
        for key in file_dict:
            f = file_dict[key]

            annotations_key = os.path.basename(f.path)

            kwargs = (
                {"references": reference_ids}
                if reference_ids is not None
                else {}
            )

            samplesets.append(
                SampleSet(
                    key=f.path,
                    reference_id=f.reference_id,
                    subject_id=f.subject_id,
                    **kwargs,
                    samples=[
                        DelayedSample(
                            key=f.path,
                            annotations=self.annotations[annotations_key],
                            load=partial(
                                bob.io.base.load,
                                os.path.join(
                                    self.original_directory,
                                    f.path + self.extension,
                                ),
                            ),
                        )
                    ],
                )
            )
        return samplesets
