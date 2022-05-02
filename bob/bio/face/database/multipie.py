#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  Multipie database implementation
"""

from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.bio.face.database.sample_loaders import MultiposeAnnotations
from bob.extension import rc
from bob.extension.download import get_file


class MultipieDatabase(CSVDataset):
    """

    The `CMU Multi-PIE face database <http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html>`_ contains more than 750,000 images
    of 337 people recorded in up to four sessions over the span of five months. Subjects were imaged under 15 view points and 19 illumination
    conditions while displaying a range of facial expressions. In addition, high resolution frontal images were acquired as well.
    In total, the database contains more than 305 GB of face data.

    The data has been recorded over 4 sessions. For each session, the subjects were asked to display a few
    different expressions. For each of those expressions, a complete set of 30 pictures is captured that includes
    15 different view points times 20 different illumination conditions (18 with various flashes, plus 2 pictures with no flash at all).


    .. warning::

      To use this dataset protocol, you need to have the original files of the Multipie dataset.
      Once you have it downloaded, please run the following command to set the path for Bob

        .. code-block:: sh

            bob config set bob.db.multipie.directory [MULTIPIE PATH]


    Available expressions:

     - Session 1 : *neutral*, *smile*
     - Session 2 : *neutral*, *surprise*, *squint*
     - Session 3 : *neutral*, *smile*, *disgust*
     - Session 4 : *neutral*, *neutral*, *scream*.

    Camera and flash positioning:

    The different view points are obtained by a set of 13 cameras located at head height, spaced at 15° intervals,
    from the -90° to the 90° angle, plus 2 additional cameras located above the subject to simulate a typical
    surveillance view. A flash coincides with each camera, and 3 additional flashes are positioned above the subject, for a total
    of 18 different possible flashes.

    Protocols:

    **Expression protocol**

    **Protocol E**

    * Only frontal view (camera 05_1); only no-flash (shot 0)
    * Enrolled : 1x neutral expression (session 1; recording 1)
    * Probes : 4x neutral expression + other expressions (session 2, 3, 4; all recordings)

    **Pose protocol**

    **Protocol P**

    * Only neutral expression (recording 1 from each session, + recording 2 from session 4); only no-flash (shot 0)
    * Enrolled : 1x frontal view (session 1; camera 05_1)
    * Probes : all views from cameras at head height (i.e excluding 08_1 and 19_1), including camera 05_1 from session 2,3,4.

    **Illumination protocols**

    N.B : shot 19 is never used in those protocols as it is redundant with shot 0 (both are no-flash).

    **Protocol M**

    * Only frontal view (camera 05_1); only neutral expression (recording 1 from each session, + recording 2 from session 4)
    * Enrolled : no-flash (session 1; shot 0)
    * Probes : no-flash (session 2, 3, 4; shot 0)

    **Protocol U**

    * Only frontal view (camera 05_1); only neutral expression (recording 1 from each session, + recording 2 from session 4)
    * Enrolled : no-flash (session 1; shot 0)
    * Probes : all shots from session 2, 3, 4, including shot 0.

    **Protocol G**

    * Only frontal view (camera 05_1); only neutral expression (recording 1 from each session, + recording 2 from session 4)
    * Enrolled : all shots (session 1; all shots)
    * Probes : all shots from session 2, 3, 4.




    """

    def __init__(
        self, protocol, annotation_type="eyes-center", fixed_positions=None
    ):

        # Downloading model if not exists
        urls = MultipieDatabase.urls()
        filename = get_file(
            "multipie.tar.gz",
            urls,
            file_hash="6c27c9616c2d0373c5f052b061d80178",
        )

        super().__init__(
            name="multipie",
            dataset_protocol_path=filename,
            protocol=protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc["bob.db.multipie.directory"]
                    if rc["bob.db.multipie.directory"]
                    else "",
                    extension=".png",
                ),
                MultiposeAnnotations(),
            ),
            annotation_type=["eyes-center", "left-profile", "right-profile"],
            fixed_positions=fixed_positions,
        )

    @staticmethod
    def protocols():
        # TODO: Until we have (if we have) a function that dumps the protocols, let's use this one.
        return [
            "P240",
            "P191",
            "P130",
            "G",
            "P010",
            "P041",
            "P051",
            "P050",
            "M",
            "P110",
            "P",
            "P140",
            "U",
            "P200",
            "E",
            "P190",
            "P120",
            "P080",
            "P081",
            "P090",
        ]

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/multipie_v2.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/multipie_v2.tar.gz",
        ]
