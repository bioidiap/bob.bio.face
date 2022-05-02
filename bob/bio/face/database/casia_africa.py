#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  CASIA-Face-Africa: database implementation
"""

from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.extension import rc
from bob.extension.download import get_file


class CasiaAfricaDatabase(CSVDataset):
    """
    The Casia-Face-Africa dataset is composed of 1133 identities from different ethical groups in Nigeria.

    The capturing locations are:
      - Dabai  city  in  Katsina  state
      - Hotoro  in  Kano  state
      - Birget  in Kano  state
      - Gandun  Albasa  in  Kano  state
      - Sabon  Gari  inKano  state
      - Kano  State  School  of  Technology

    These locations were strategically selected as they are known to have diverse population of local ethnicities.

    .. warning::
       Only 17 subjects had their images capture in two sessions.

    Images were captured during  daytime  and  night using three different cameras:
      - C1: Visual Light Camera
      - C2: Visual Light Camera
      - C3: NIR camera


    This dataset interface implemented the three verificatio protocols: "ID-V-All-Ep1", "ID-V-All-Ep2", and "ID-V-All-Ep3"
    and they are organized as the following:

    +------------------------------------------------------------------------------------+
    |                     Dev. Set                                                       |
    +------------------+----------------------------+------------+----------+------------+
    |  protocol name   |   Cameras (gallery/probe)  | Identities | Gallery  | Probes     |
    +==================+============================+============+==========+============+
    | ID-V-All-Ep1     |      C1/C2                 |   1133     |   2455   |   2426     |
    +------------------+----------------------------+------------+----------+------------+
    | ID-V-All-Ep2     |      C1/C3                 |   1133     |   2455   |   1171     |
    +------------------+----------------------------+------------+----------+------------+
    | ID-V-All-Ep3     |      C2/C3                 |   1133     |   2466   |   1193     |
    +------------------+----------------------------+------------+----------+------------+


    .. warning::
        Use the command below to set the path of the real data::

            $ bob config set bob.db.casia-africa.directory [PATH-TO-MEDS-DATA]


    .. code-block:: latex

        @article{jawad2020,
           author  =  {Jawad,  Muhammad  and  Yunlong,  Wang  andCaiyong,  Wang  and  Kunbo,  Zhang  and Zhenan, Sun},
           title = {CASIA-Face-Africa: A Large-scale African Face Image Database},
           journal = {IEEE Transactions on Information Forensics and Security},
           pages = {},
           ISSN = {},
           year = {},
           type = {Journal Article}
        }


    Example
    -------

    Fetching biometric references::

    >>> from bob.bio.face.database import CasiaAfricaDatabase
    >>> database = CasiaAfricaDatabase(protocol="ID-V-All-Ep1")
    >>> database.references()


    Fetching probes::

    >>> from bob.bio.face.database import CasiaAfricaDatabase
    >>> database = CasiaAfricaDatabase(protocol="ID-V-All-Ep1")
    >>> database.probes()


    Parameters
    ----------

    protocol: str
        One of the database protocols. Options are "ID-V-All-Ep1", "ID-V-All-Ep2" and "ID-V-All-Ep3"
    """

    def __init__(
        self, protocol, annotation_type="eyes-center", fixed_positions=None
    ):

        # Downloading model if not exists
        urls = CasiaAfricaDatabase.urls()
        filename = get_file(
            "casia-africa.tar.gz",
            urls,
            file_hash="080d4bfffec95a6445507065054757eb",
        )

        directory = (
            rc["bob.db.casia-africa.directory"]
            if rc["bob.db.casia-africa.directory "]
            else ""
        )

        super().__init__(
            name="casia-africa",
            dataset_protocol_path=filename,
            protocol=protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=directory,
                    extension=".jpg",
                    reference_id_equal_subject_id=False,
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
            "ID-V-All-Ep1",
            "ID-V-All-Ep2",
            "ID-V-All-Ep3",
        ]

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/casia-africa.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/casia-africa.tar.gz",
        ]
