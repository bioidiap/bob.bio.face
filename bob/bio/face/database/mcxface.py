#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Anjith George <anjith.george@idiap.ch>

"""
  MCXFace database implementation 
"""

from bob.bio.base.database import CSVDataset
from bob.bio.base.database import CSVToSampleLoaderBiometrics
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.extension import rc
from bob.extension.download import get_file
import bob.io.base
from sklearn.pipeline import make_pipeline


class MCXFaceDatabase(CSVDataset):
    """
    Collected at Idiap for the BATL ODIN project, the MCXFace is derived from the HQ-WMCA dataset.
    The database implements several Face Recognition 
    protocols, ie between the same modalities and heterogeneous face recognition as well. 
    The database has only the train and dev splits due to the limited number of subjects.
    A total of 51 subjects are present in the dataset collected across several sessions. 30 subjects are in 
    train fold and 20 subjects are in dev fold. 
    The dataset contains the following channels:

    COLOR: From Basler Camera
    DEPTH: From Intel D415
    THERMAL: From Xenics Gobi   
    NIR: Is again from Basler NIR camera, we use wavelength 850nm
    SWIR: From Xenics Bobcat we use only 1300nm wavelength.

    All the channels are spatially and temporally registered meaning, one
    can share the annotations provided between channels. The left and right
    eye centers are provided as annotations.

    The protocols are as follows: SOURCE_TARGET_split, where the SOURCE
    is the modality used for enrollment, and TARGET is the modality used 
    as probes. We implement several combinations as the protocols. In addition,
    we have normal recognition protocols where both source and target are the same 
    modalities. For each of these, we have also prepared 5 different splits by 
    randomly splitting the clients between train and dev sets. Subjects who have 
    only one session are always assigned to the training fold.

    .. warning::
        Use the command below to set the path of the real data::

            $ bob config set bob.db.mcxface.directory [PATH-TO-MCXFACE-DATA]



    Parameters
    ----------

    protocol: str
        One of the database protocols.
    """

    def __init__(self, protocol):

        # Downloading model if not exists
        urls = MCXFaceDatabase.urls()
        filename = get_file(
            "mcxface.tar.gz", urls, file_hash="c4b73aa7cee7dc2b9bfc2b20d48db5b8",
        )

        self.annotation_type = "eyes-center"
        self.fixed_positions = None

        directory = (
            rc["bob.db.mcxface.directory"]
            if rc["bob.db.mcxface.directory"]
            else ""
        )

        def load(path):
            """
            Images in this dataset are stored as 8-bit jpg 
            """
            return bob.io.base.load(path) 

        super().__init__(
            filename,
            protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=load,
                    dataset_original_directory=directory,
                    extension=".jpg",
                ),
                EyesAnnotations(),
            ),
        )

    @staticmethod
    def protocols():
        return ['COLOR-COLOR-split1',
         'DEPTH-DEPTH-split1',
         'THERMAL-THERMAL-split1',
         'SWIR-SWIR-split1',
         'NIR-NIR-split1',
         'COLOR-DEPTH-split1',
         'COLOR-THERMAL-split1',
         'COLOR-SWIR-split1',
         'COLOR-NIR-split1',
         'COLOR-COLOR-split2',
         'DEPTH-DEPTH-split2',
         'THERMAL-THERMAL-split2',
         'SWIR-SWIR-split2',
         'NIR-NIR-split2',
         'COLOR-DEPTH-split2',
         'COLOR-THERMAL-split2',
         'COLOR-SWIR-split2',
         'COLOR-NIR-split2',
         'COLOR-COLOR-split3',
         'DEPTH-DEPTH-split3',
         'THERMAL-THERMAL-split3',
         'SWIR-SWIR-split3',
         'NIR-NIR-split3',
         'COLOR-DEPTH-split3',
         'COLOR-THERMAL-split3',
         'COLOR-SWIR-split3',
         'COLOR-NIR-split3',
         'COLOR-COLOR-split4',
         'DEPTH-DEPTH-split4',
         'THERMAL-THERMAL-split4',
         'SWIR-SWIR-split4',
         'NIR-NIR-split4',
         'COLOR-DEPTH-split4',
         'COLOR-THERMAL-split4',
         'COLOR-SWIR-split4',
         'COLOR-NIR-split4',
         'COLOR-COLOR-split5',
         'DEPTH-DEPTH-split5',
         'THERMAL-THERMAL-split5',
         'SWIR-SWIR-split5',
         'NIR-NIR-split5',
         'COLOR-DEPTH-split5',
         'COLOR-THERMAL-split5',
         'COLOR-SWIR-split5',
         'COLOR-NIR-split5']


    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/~ageorge/mcxface.tar.gz",
            "https://www.idiap.ch/~ageorge/mcxface.tar.gz",
        ]
