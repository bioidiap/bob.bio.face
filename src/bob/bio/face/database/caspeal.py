#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  Caspeal database implementation
"""

from clapper.rc import UserDefaults
from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDatabase, FileSampleLoader
from bob.bio.face.database.sample_loaders import EyesAnnotations

rc = UserDefaults("bobrc.toml")


class CaspealDatabase(CSVDatabase):
    """
    The CAS-PEAL database consists of several ten thousand images of Chinese people (CAS = Chinese Academy of Science).
    Overall, there are 1040 identities contained in the database.
    For these identities, images with different Pose, Expression, Aging and Lighting (PEAL) conditions, as well as accessories, image backgrounds and camera distances are provided.

    Included in the database, there are file lists defining identification experiments.
    All the experiments rely on a gallery that consists of the frontal and frontally illuminated images with neutral expression and no accessories.
    For each of the variations, probe sets including exactly that variation are available.

    The training set consists of a subset of the frontal images (some images are both in the training and in the development set).
    This also means that there is no training set defined for the pose images.
    Additionally, the database defines only a development set, but no evaluation set.

    This package only contains the Bob_ accessor methods to use the DB directly from python, with our certified protocols.
    We have implemented the default face identification protocols ``'accessory'``, ``'aging'``, ``'background'``, ``'distance'``, ``'expression'`` and ``'lighting'``.
    We do not provide the ``'pose'`` protocol (yet) since the training set of the `CAS-PEAL <http://www.jdl.ac.cn/peal/files/IEEE_SMC_A_gao_CAS-PEAL.pdf>`_
    database does not contain pose images:


    .. code-block:: latex

        @article{gao2007cas,
        title={The CAS-PEAL large-scale Chinese face database and baseline evaluations},
        author={Gao, Wen and Cao, Bo and Shan, Shiguang and Chen, Xilin and Zhou, Delong and Zhang, Xiaohua and Zhao, Debin},
        journal={IEEE Transactions on Systems, Man, and Cybernetics-Part A: Systems and Humans},
        volume={38},
        number={1},
        pages={149--161},
        year={2007},
        publisher={IEEE}
        }

    """

    name = "caspeal"
    category = "face"
    dataset_protocols_name = "caspeal.tar.gz"
    dataset_protocols_urls = [
        "https://www.idiap.ch/software/bob/databases/latest/face/caspeal-9ce68f00.tar.gz",
        "http://www.idiap.ch/software/bob/databases/latest/face/caspeal-9ce68f00.tar.gz",
    ]

    dataset_protocols_hash = "9ce68f00"

    def __init__(
        self, protocol, annotation_type="eyes-center", fixed_positions=None
    ):
        super().__init__(
            name=self.name,
            protocol=protocol,
            transformer=make_pipeline(
                FileSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc.get(
                        "bob.bio.face.caspeal.directory", ""
                    ),
                    extension=rc.get("bob.bio.face.caspeal.extension", ".tif"),
                ),
                EyesAnnotations(),
            ),
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
        )
