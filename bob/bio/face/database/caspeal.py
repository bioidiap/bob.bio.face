#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  Multipie database implementation
"""

from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.extension import rc
from bob.extension.download import get_file


class CaspealDatabase(CSVDataset):
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

    def __init__(
        self, protocol, annotation_type="eyes-center", fixed_positions=None
    ):

        # Downloading model if not exists
        urls = CaspealDatabase.urls()
        filename = get_file(
            "caspeal.tar.gz",
            urls,
            file_hash="1c77f660ef85fa263a2312fd8263d0d9",
        )

        super().__init__(
            name="caspeal",
            dataset_protocol_path=filename,
            protocol=protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc[
                        "bob.bio.face.caspeal.directory"
                    ]
                    if rc["bob.bio.face.caspeal.directory"]
                    else "",
                    extension=".png",
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
            "accessory",
            "aging",
            "background",
            "distance",
            "expression",
            "lighting",
        ]

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/caspeal.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/caspeal.tar.gz",
        ]
