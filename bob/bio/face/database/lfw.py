#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2016

from bob.bio.base.pipelines.vanilla_biometrics.abstract_classes import Database
from bob.pipelines import DelayedSample, SampleSet
import os
from functools import partial
from bob.extension import rc
import bob.io.image
import copy

import logging
import numpy as np

logger = logging.getLogger(__name__)


class LFWDatabase(Database):
    """
    This package contains the access API and descriptions for the `Labeled Faced in the Wild <http://vis-www.cs.umass.edu/lfw>`_ (LFW) database.
    It only contains the Bob_ accessor methods to use the DB directly from python, with our certified protocols.
    The actual raw data for the database should be downloaded from the original URL (though we were not able to contact the corresponding Professor).


    The LFW database provides two different sets (called "views").
    The first one, called ``view1`` is used for optimizing meta-parameters of your algorithm.
    The second one, called ``view2`` is used for benchmarking.
    This interface supports only the ``view2`` protocol.
    Please note that in ``view2`` there is only a ``'dev'`` group, but no ``'eval'``.


    .. warning::
      
      To use this dataset protocol, you need to have the original files of the LFW datasets.
      Once you have it downloaded, please run the following command to set the path for Bob

        .. code-block:: sh

            bob config set bob.bio.face.lfw.directory [LFW PATH]
            bob config set bob.bio.face.lfw.annotation_directory [LFW ANNOTATION_PATH] # for the annotations



    .. code-block:: python

        >>> from bob.bio.face.database import LFWDatabase
        >>> lfw = LFWDatabase(protocol="view2")
        >>>
        >>> # Fetching the gallery 
        >>> references = lfw.references()
        >>> # Fetching the probes 
        >>> probes = lfw.probes()

    
    
    Parameters
    ----------

      protocol: str
        One of the database protocols. Options are `view2`

      annotation_type: str
        Type of the annotations used for face crop. Default to `eyes-center`

      image_relative_path: str
        LFW provides several types image crops. Some with the full image, some with with specific
        face crop. Use this variable to set which image crop you want. Default to `all_images`, which means
        no crop.

      annotation_directory: str
        LFW annotations path. Default to what is set in the variable `bob.bio.face.lfw.directory`

      original_directory: str
        LFW phisical path. Default to what is set in the variable `bob.bio.face.lfw.directory`

      annotation_issuer: str
        Type of the annotations. Default to `funneled`. Possible types `funneled` or `idiap`

    """

    def __init__(
        self,
        protocol,
        annotation_type="eyes-center",
        image_relative_path="all_images",
        fixed_positions=None,
        original_directory=rc.get("bob.bio.face.lfw.directory"),
        extension=".jpg",
        annotation_directory=rc.get("bob.bio.face.lfw.annotation_directory"),
        annotation_issuer="funneled",
    ):

        if original_directory is None or not os.path.exists(original_directory):
            raise ValueError(
                "Invalid or non existant `original_directory`: f{original_directory}."
                "Please, do `bob config set bob.bio.face.lfw.directory PATH` to set the LFW data directory."
            )

        if annotation_directory is None or not os.path.exists(annotation_directory):
            logger.warning(
                "Invalid or non existant `annotation_directory`: f{annotation_directory}."
                "As a result, `SampleSet` will not contain annotations"
                "Please, do `bob config set bob.bio.face.lfw.annotation_directory PATH` to set the LFW annotation directory."
            )

        if annotation_issuer != "funneled" and annotation_issuer != "idiap":
            raise ValueError(
                f"Invalid annotation issuer: {annotation_issuer}. Possible values are `idiap` or `funneled`"
            )
        self.annotation_issuer = annotation_issuer
        # Hard-coding the extension of the annotations
        # I don't think we need this exposed
        # Please, open an issue if otherwise
        self.annotation_extension = (
            ".pos" if annotation_issuer == "idiap" else ".jpg.pts"
        )

        self._check_protocol(protocol)

        self.references_dict = {}
        self.probes_dict = {}
        self.pairs = {}
        self.probe_reference_keys = {}  # Inverted pairs

        self.annotations = None
        self.original_directory = original_directory
        self.annotation_directory = annotation_directory
        self.extension = extension
        self.image_relative_path = image_relative_path

        # Some path manipulation lambdas
        self.subject_id_from_filename = lambda x: "_".join(x.split("_")[0:-1])

        self.make_path_from_filename = lambda x: os.path.join(
            self.subject_id_from_filename(x), x
        )

        self.load_pairs()

        super().__init__(
            name="lfw",
            protocol=protocol,
            allow_scoring_with_all_biometric_references=False,
            annotation_type=annotation_type,
            fixed_positions=None,
            memory_demanding=False,
        )

    def _extract_funneled(self, annotation_path):
        """Interprets the annotation string as if it came from the funneled images.
        Inspired by: https://gitlab.idiap.ch/bob/bob.db.lfw/-/blob/5ac22c5b77aae971de6b73cbe23f26d6a5632072/bob/db/lfw/models.py#L69
        """
        with open(annotation_path) as f:
            splits = np.array(f.readlines()[0].split(" "), "float")

        assert len(splits) == 18
        locations = [
            "reyeo",
            "reyei",
            "leyei",
            "leyeo",
            "noser",
            "noset",
            "nosel",
            "mouthr",
            "mouthl",
        ]
        annotations = dict(
            [
                (locations[i], (float(splits[2 * i + 1]), float(splits[2 * i])))
                for i in range(9)
            ]
        )
        # add eye center annotations as the center between the eye corners
        annotations["leye"] = (
            (annotations["leyei"][0] + annotations["leyeo"][0]) / 2.0,
            (annotations["leyei"][1] + annotations["leyeo"][1]) / 2.0,
        )
        annotations["reye"] = (
            (annotations["reyei"][0] + annotations["reyeo"][0]) / 2.0,
            (annotations["reyei"][1] + annotations["reyeo"][1]) / 2.0,
        )

        return annotations

    def _extract_idiap(self, annotation_file):
        """Interprets the annotation string as if it came from the Idiap annotations.
        Inspired by: https://gitlab.idiap.ch/bob/bob.db.lfw/-/blob/5ac22c5b77aae971de6b73cbe23f26d6a5632072/bob/db/lfw/models.py#L81"""

        annotations = {}
        splits = {}
        with open(annotation_path) as f:
            for line in f.readlines():
                line = line.split(" ")
                if len(line) == 3:
                    # splits.append([float(line[2]), float(line[1])])
                    splits[int(line[0])] = (float(line[1]), float(line[2]))

        if 3 in splits:
            annotations["reye"] = splits[3]

        if 8 in splits:
            annotations["leye"] = splits[8]

        return annotations

    def load_pairs(self):
        pairs_path = os.path.join(self.original_directory, "view2", "pairs.txt")
        self.pairs = {}

        make_filename = lambda name, index: f"{name}_{index.zfill(4)}"

        with open(pairs_path) as f:
            for i, line in enumerate(f.readlines()):
                # Skip the first line
                if i == 0:
                    continue

                line = line.split("\t")

                # Three lines, genuine pairs otherwise impostor
                if len(line) == 3:
                    # self.subject_id_from_filename()
                    key_filename = make_filename(line[0], line[1].rstrip("\n"))
                    value_filename = make_filename(line[0], line[2].rstrip("\n"))

                else:
                    key_filename = make_filename(line[0], line[1].rstrip("\n"))
                    value_filename = make_filename(line[2], line[3].rstrip("\n"))

                key = self.make_path_from_filename(key_filename)
                value = self.make_path_from_filename(value_filename)

                if key not in self.pairs:
                    self.pairs[key] = []
                self.pairs[key].append(value)

        self._create_probe_reference_dict()

    @staticmethod
    def protocols():
        return ["view2"]

    def background_model_samples(self):
        return []

    def _create_probe_reference_dict(self):
        """
        Returns a dictionary whose each key (probe key) holds the list of biometric references
        where that probe should be compared with.
        """

        self.probe_reference_keys = {}
        for key in self.pairs:
            for value in self.pairs[key]:

                if value not in self.probe_reference_keys:
                    self.probe_reference_keys[value] = []

                self.probe_reference_keys[value].append(key)

    def probes(self, group="dev"):
        if self.protocol not in self.probes_dict:
            self.probes_dict[self.protocol] = []

            for key in self.probe_reference_keys:
                image_path = os.path.join(
                    self.original_directory,
                    self.image_relative_path,
                    key + self.extension,
                )
                if self.annotation_directory is not None:
                    annotation_path = os.path.join(
                        self.annotation_directory, key + self.annotation_extension,
                    )
                    annotations = (
                        self._extract_funneled(annotation_path)
                        if self.annotation_issuer == "funneled"
                        else self._extract_idiap(annotation_path)
                    )
                else:
                    annotations = None

                sset = SampleSet(
                    key=key,
                    reference_id=key,
                    subject_id=self.subject_id_from_filename(key),
                    references=copy.deepcopy(
                        self.probe_reference_keys[key]
                    ),  # deep copying to avoid bizarre issues with dask
                    samples=[
                        DelayedSample(
                            key=key,
                            load=partial(bob.io.image.load, image_path),
                            annotations=annotations,
                        )
                    ],
                )
                self.probes_dict[self.protocol].append(sset)

        return self.probes_dict[self.protocol]

    def references(self, group="dev"):

        if self.protocol not in self.references_dict:
            self.references_dict[self.protocol] = []

            for key in self.pairs:

                image_path = os.path.join(
                    self.original_directory,
                    self.image_relative_path,
                    key + self.extension,
                )
                if self.annotation_directory is not None:
                    annotation_path = os.path.join(
                        self.annotation_directory, key + self.annotation_extension,
                    )
                    annotations = (
                        self._extract_funneled(annotation_path)
                        if self.annotation_issuer == "funneled"
                        else self._extract_idiap(annotation_path)
                    )
                else:
                    annotations = None

                sset = SampleSet(
                    key=key,
                    reference_id=key,
                    subject_id=self.subject_id_from_filename(key),
                    samples=[
                        DelayedSample(
                            key=key,
                            load=partial(bob.io.image.load, image_path),
                            annotations=annotations,
                        )
                    ],
                )
                self.references_dict[self.protocol].append(sset)

        return self.references_dict[self.protocol]

    def groups(self):
        return ["dev"]

    def all_samples(self, group="dev"):
        self._check_group(group)

        return self.references() + self.probes()

    def _check_protocol(self, protocol):
        assert protocol in self.protocols(), "Unvalid protocol `{}` not in {}".format(
            protocol, self.protocols()
        )

    def _check_group(self, group):
        assert group in self.groups(), "Unvalid group `{}` not in {}".format(
            group, self.groups()
        )
