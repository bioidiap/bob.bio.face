#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  CBSRNirVis2Database database: database implementation
"""

import os

from clapper.rc import UserDefaults
from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDatabase, FileSampleLoader
from bob.bio.face.database.sample_loaders import EyesAnnotations

rc = UserDefaults("bobrc.toml")


class CBSRNirVis2Database(CSVDatabase):
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

    name = "cbsr-nir-vis2"
    category = "face"
    dataset_protocols_name = "cbsr-nir-vis2.tar.gz"
    dataset_protocols_urls = [
        "https://www.idiap.ch/software/bob/databases/latest/face/cbsr-nir-vis2-cabebf97.tar.gz",
        "http://www.idiap.ch/software/bob/databases/latest/face/cbsr-nir-vis2-cabebf97.tar.gz",
    ]
    dataset_protocols_hash = "cabebf97"

    def __init__(
        self, protocol, annotation_type="eyes-center", fixed_positions=None
    ):
        def load(filename):
            extensions = [".jpg", ".bmp"]
            for e in extensions:
                f = os.path.splitext(filename)[0]
                new_filename = f + e
                if os.path.exists(new_filename):
                    return bob.io.base.load(new_filename)
            raise ValueError("File `{0}` not found".format(str(new_filename)))

        super().__init__(
            name="",
            protocol=protocol,
            transformer=make_pipeline(
                FileSampleLoader(
                    data_loader=load,
                    dataset_original_directory=rc.get(
                        "bob.db.cbsr-nir-vis-2.directory", ""
                    ),
                    extension=rc.get("bob.db.cbsr-nir-vis-2.extension", ".jpg"),
                ),
                EyesAnnotations(),
            ),
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
        )
