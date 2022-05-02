.. vim: set fileencoding=utf-8 :

.. _bob.bio.face.leaderboard.multipie:

================
Multipie Dataset
================

.. todo::
   Benchmarks on Multipie Database

   Probably for Manuel's students

Database
========
The `CMU Multi-PIE face database <http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html>`_ contains more than 750,000 images
of 337 people recorded in up to four sessions over the span of five months. Subjects were imaged under 15 view points and 19 illumination
conditions while displaying a range of facial expressions. In addition, high resolution frontal images were acquired as well.
In total, the database contains more than 305 GB of face data.

Content
*******
The data has been recorded over 4 sessions. For each session, the subjects were asked to display a few
different expressions. For each of those expressions, a complete set of 30 pictures is captured that includes
15 different view points times 20 different illumination conditions (18 with various flashes, plus 2 pictures with no flash at all).

Available expressions
---------------------
* Session 1 : *neutral*, *smile*
* Session 2 : *neutral*, *surprise*, *squint*
* Session 3 : *neutral*, *smile*, *disgust*
* Session 4 : *neutral*, *neutral*, *scream*.

Camera and flash positioning
----------------------------
The different view points are obtained by a set of 13 cameras located at head height, spaced at 15° intervals,
from the -90° to the 90° angle, plus 2 additional cameras located above the subject to simulate a typical
surveillance view. A flash coincides with each camera, and 3 additional flashes are positioned above the subject, for a total
of 18 different possible flashes.

The following picture showcase the positioning of the cameras (in yellow) and of the flashes (in white).

.. image:: img/multipie/multipie_setup.jpg
    :width: 620px
    :align: center
    :height: 200px
    :alt: Multipie setup

File paths
----------

The data directory structure and filenames adopt the following structure:

.. code-block:: shell

   session<XX>/multiview/<subject_id>/<recording_id>/<camera_id>/<subject_id>_<session_id>_<recording_id>_<camera_id>_<shot_id>.png

For example, the file

.. code-block:: shell

   session02/multiview/001/02/05_1/001_02_02_051_07.png

corresponds to
* Subject 001
* Session 2
* Second recording -> Expression is *surprise*
* Camera 05_1 -> Frontal view
* Shot 07 -> Illumination through the frontal flash

Protocols
*********

Expression protocol
-------------------
**Protocol E**

* Only frontal view (camera 05_1); only no-flash (shot 0)
* Enrolled : 1x neutral expression (session 1; recording 1)
* Probes : 4x neutral expression + other expressions (session 2, 3, 4; all recordings)

Pose protocol
-------------
**Protocol P**

* Only neutral expression (recording 1 from each session, + recording 2 from session 4); only no-flash (shot 0)
* Enrolled : 1x frontal view (session 1; camera 05_1)
* Probes : all views from cameras at head height (i.e excluding 08_1 and 19_1), including camera 05_1 from session 2,3,4.

Illumination protocols
----------------------
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


Setting up the database
=======================

    To use this dataset protocol, you need to have the original files of the MultiPIE dataset.
    Once you have it downloaded, please run the following command to set the path for Bob

    .. code-block:: sh

        bob config set bob.db.multipie.directory [MULTIPIE PATH]



Benchmarking
============


Running experiments
*******************


You can run the Multipie baselines command with a simple command such as:

.. code-block:: bash

    bob bio pipeline simple multipie iresnet100 -m -l sge

Note that the default protocol implemented in the resource is the U protocol.
The pose protocol is also available using

.. code-block:: bash

    bob bio pipeline simple multipie_pose iresnet100 -m -l sge

For the other protocols, one has to define its own configuration file (e.g.: *multipie_M.py*) as follows:

.. code-block:: python

    from bob.bio.face.database import MultipieDatabase
    database = MultipieDatabase(protocol="M")

then point to it when calling the pipeline execution:

.. code-block:: bash

    bob bio pipeline simple multipie_M.py iresnet100 -m -l sge



Plots
*****

:ref:`bob.bio.face` has some customized plots where the FMR and FNMR trade-off in the evaluation set can be plot using operational
FMR thresholds from the development set.
This is done be the command `bob bio face plots multipie-pose` command as in the example below:


Pose protocol
-------------

.. code-block:: bash

   wget https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/scores/frice_scores.tar.gz
   tar -xzvf frice_scores.tar.gz
   bob bio face plots multipie-pose -e \
        ./frice_scores/multipie-pose/arcface_insightface/scores-{dev,eval}.csv \
        ./frice_scores/multipie-pose/iresnet50_msceleb_arcface_20210623/scores-{dev,eval}.csv \
        ./frice_scores/multipie-pose/attention_net/scores-{dev,eval}.csv \
        ./frice_scores/multipie-pose/facenet_sanderberg/scores-{dev,eval}.csv \
        ./frice_scores/multipie-pose/ISV/scores-{dev,eval}.csv \
        --titles ArcFace-100,Idiap-Resnet50,Zoo-AttentionNet,Facenet-Sandberg,2014-ISV -o plot.pdf


Expression protocol
-------------------

.. code-block:: bash

    bob bio face plots multipie-expression -e \
        ./frice_scores/multipie-expression/arcface_insightface/scores-{dev,eval}.csv \
        ./frice_scores/multipie-expression/iresnet50_msceleb_arcface_20210623/scores-{dev,eval}.csv \
        ./frice_scores/multipie-expression/attention_net/scores-{dev,eval}.csv \
        ./frice_scores/multipie-expression/facenet_sanderberg/scores-{dev,eval}.csv \
        ./frice_scores/multipie-expression/ISV/scores-{dev,eval}.csv \
        --titles ArcFace-100,Idiap-Resnet50,Zoo-AttentionNet,Facenet-Sandberg,2014-ISV -o plot.pdf

.. note::
  Always remember, `bob bio face plots --help` is your friend.
