#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  CBSRNirVis2Database database: database implementation
"""

import os

from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.extension import rc
from bob.extension.download import get_file


class CBSRNirVis2Database(CSVDataset):
    """
    This package contains the access API and descriptions for the `CASIA NIR-VIS 2.0 Database <http://www.cbsr.ia.ac.cn/english/NIR-VIS-2.0-Database.html>`.
    The actual raw data for the database should be downloaded from the original URL.
    This package only contains the Bob accessor methods to use the DB directly from python, with the original protocol of the database.

    CASIA NIR-VIS 2.0 database offers pairs of mugshot images and their correspondent NIR photos.
    The images of this database were collected in four recording sessions: 2007 spring, 2009 summer, 2009 fall and 2010 summer,
    in which the first session is identical to the CASIA HFB database. It consists of 725 subjects in total.
    There are [1-22] VIS and [5-50] NIR face images per subject. The eyes positions are also distributed with the images.


    .. code-block:: latex

        @inproceedings{li2013casia,
        title={The casia nir-vis 2.0 face database},
        author={Li, Stan Z and Yi, Dong and Lei, Zhen and Liao, Shengcai},
        booktitle={Computer Vision and Pattern Recognition Workshops (CVPRW), 2013 IEEE Conference on},
        pages={348--353},
        year={2013},
        organization={IEEE}
        }


    .. warning::
        Use the command below to set the path of the real data::

            $ bob config set bob.db.cbsr-nir-vis-2.directory [PATH-TO-CBSR-DATA]



    Parameters
    ----------

    protocol: str
        One of the database protocols.
    """

    def __init__(
        self, protocol, annotation_type="eyes-center", fixed_positions=None
    ):

        # Downloading model if not exists
        urls = CBSRNirVis2Database.urls()
        filename = get_file(
            "cbsr-nir-vis2.tar.gz",
            urls,
            file_hash="e4bda52ab6754556783d6730eccc2ae2",
        )

        directory = (
            rc["bob.db.cbsr-nir-vis-2.directory"]
            if rc["bob.db.cbsr-nir-vis-2.directory"]
            else ""
        )

        def load(filename):
            extensions = [".jpg", ".bmp"]
            for e in extensions:
                f = os.path.splitext(filename)[0]
                new_filename = f + e
                if os.path.exists(new_filename):
                    return bob.io.base.load(new_filename)
            else:
                raise ValueError(
                    "File `{0}` not found".format(str(new_filename))
                )

        super().__init__(
            name="cbsr-nir-vis2",
            dataset_protocol_path=filename,
            protocol=protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=load,
                    dataset_original_directory=directory,
                    extension=".jpg",
                ),
                EyesAnnotations(),
            ),
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
        )

    @staticmethod
    def protocols():
        # TODO: Until we have (if we have) a function that dumps the protocols, let's use this one.
        return [
            "view2_1",
            "view2_2",
            "view2_3",
            "view2_4",
            "view2_5",
            "view2_6",
            "view2_7",
            "view2_8",
            "view2_9",
            "view2_10",
        ]

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/cbsr-nir-vis2.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/cbsr-nir-vis2.tar.gz",
        ]
