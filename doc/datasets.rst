.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <siebenkopf@googlemail.com>

.. _bob.bio.face.datasets:

Setting up Facial Image Databases
=================================

This package includes a large list of facial image datasets including their default evaluation protocols.
You can list all available datasets and protocols by calling:

.. code-block:: bash

  resources.py --types databases


To use one of the available databases in an experiment, we need to tell the system where the original files can be located.
We use the default bob configuration system for setting this us, which basically translates to:

.. code-block:: bash

  bob config set <key> <value>

Follow below an example using this command to set the database path for one of the supported datasets.

.. code-block:: bash

  bob config set bob.bio.face.scface.directory [PATH_TO_SCFACE_DATASET]


In some circumstances, the filename extension of the datasets can change from version to version, so they can be setup as well.
In the following, find a list of all current databases, their configuration parameters, and their provided protocols, in alphabetic order.

* `ARface dataset <https://www2.ece.ohio-state.edu/~aleix/ARdatabase.html>`__ (:any:`bob.bio.face.database.ARFaceDatabase`)

  - directory key: ``bob.bio.face.database.arface.directory``; the directory containing all the images
  - extension key: ``bob.bio.face.database.arface.extension``; ``".ppm"`` if not set
  - protocols: ``expression, illumination, occlusion, occlusion_and_illumination, all``
  - annotations: eye centers (provided in the interface)
  - notes: The original files are stored in a raw format and need to be converted into some image format.


* AT&T dataset of faces, previously known as the ORL dataset (:any:`bob.bio.base.database.AtntBioDatabase`)

  - annotations: none, faces are already aligned
  - notes: The dataset will be downloaded completely, including all images. No special setup required.


* Casia Africa Database (:any:`bob.bio.face.database.CasiaAfricaDatabase`)

  - directory key: ``bob.db.casia-africa.directory``; the directory containing all the images
  - expected extension: ``.jpg``
  - protocols: ``ID-V-All-Ep1, ID-V-All-Ep2, ID-V-All-Ep3``,
  - annotations: eye centers (provided in the interface)
  - notes: The data is very noisy (there are no faces in some images), and the original protocols need to be pruned.


* CASPEAL (:any:`bob.bio.face.database.CaspealDatabase`)

  - directory key: ``bob.bio.face.caspeal.directory``; the directory containing all the images
  - expected extension: ``.png``
  - protocols: ``accessory, aging, background, distance, expression, lighting``,
  - annotations: eye centers (provided in the interface)


* `CASIA NIR-VIS 2.0 <http://www.cbsr.ia.ac.cn/english/NIR-VIS-2.0-Database.html>`__ (:any:`bob.bio.face.database.CBSRNirVis2Database`)

  - directory key: ``bob.db.cbsr-nir-vis-2.directory``; the directory containing all the images
  - expected extension: ``[".jpg", ".bmp"]``
  - protocols: ``view2_1, view2_2, view2_3, view2_4, view2_5, view2_6, view2_7, view2_8, view2_9, view2_10``
  - annotations: eye centers (provided in the interface)


* Face Recognition Grand Challenge (FRGC) v2.0 (:any:`bob.bio.face.database.FRGCDatabase`)

  - directory key: ``bob.bio.face.frgc.directory``; the directory containing all the images
  - expected extension:
  - protocols: ``2.0.1, 2.0.2, 2.0.4``
  - annotations: eye centers (provided in the interface)


* The Good, The Bad and The Ugly face database (GBU) (:any:`bob.bio.face.database.GBUDatabase`)

  - directory key: ``bob.bio.face.gbu.directory``; the directory containing all the images
  - expected extension: ``.jpg``
  - protocols: ``Good, Bad, Ugly``
  - annotations: eye centers (provided in the interface)


* `IARPA Janus Benchmark C <http://www.nist.gov/programs-projects/face-challenges>`__

  - directory key: ``bob.bio.face.ijbc.directory``; the base directory containing the `images` and `protocols` folders (besides others)
  - expected extensions: ``.jpg`` and ``.png``
  - implemented protocols: ``test1, test2, test4-G1, test4-G2``
  - annotations: bounding boxes (``topleft, bottomright``); there might be several faces in one image -- you need to rely on the bounding box in order to get the correct one
  - note: The implementation relies on the availability of the protocol data. If the data directory is not given, this dataset will not be accessible.


* `Labeled Faced in the Wild <http://vis-www.cs.umass.edu/lfw>`__ (LFW) database (:any:`bob.bio.face.database.LFWDatabase`)

  - directory key: ``bob.bio.face.lfw.directory``; the directory containing all the images
  - expected extension: ``.jpg``
  - protocols:

    + ``view2`` is a combination of the 10 folds in view2; no training data is provided for this protocol
    + ``o1``, ``o2``, ``o3`` are the open-set protocols implemented in [GCR17]_

  - annotations: There are three types of eyes annotations: ``funneled, idiap, named`` (provided in the interface)
  - notes: LFW comes either as the original images, or as aligned versions. The provided annotations are valid **only for the "images aligned with funneling"**, not for the original images.


* `VGG2 Face Database <https://arxiv.org/abs/1710.08092>`__  (:any:`bob.bio.face.database.VGG2Database`)

  - directory key: ``bob.bio.face.vgg2.directory``; the directory containing all the images
  - expected extension: ``.jpg`` or set by ``bob.bio.face.vgg2.extension``
  - protocols: ``vgg2-short, vgg2-full``
  - genders: ``m`` and ``f``
  - races:  ``A, B, I, U, W, N``
  - annotations: eye centers, nose, mouth, and face bounding box (provided in the interface)


*  MEDS II (:any:`bob.bio.face.database.MEDSDatabase`)

  - directory key: ``bob.db.meds.directory``; the directory containing all the images
  - expected extension: ``.jpg``
  - protocols: ``verification_fold1, verification_fold2, verification_fold3``
  - races: Black, White
  - genders: male
  - annotations: eye centers (provided in the interface)


*  `MOBIO dataset <https://www.idiap.ch/en/dataset/mobio>`__ (:any:`bob.bio.face.database.MobioDatabase`)

  - directory key: ``bob.db.mobio.directory``; the directory containing all the images
  - extension: ``.png``
  - protocols: ``laptop1-female, laptop_mobile1-female, mobile0-female, mobile0-male-female, mobile1-male, laptop1-male, laptop_mobile1-male, mobile0-male, mobile1-female,``
  - genders: male, female
  - annotations: eye centers (provided in the interface)


*  MORPH dataset (:any:`bob.bio.face.database.MorphDatabase`)

  - directory key: ``bob.db.morph.directory``; the directory containing all the images
  - extension: ``.JPG``
  - protocols: ``verification_fold1, verification_fold2, verification_fold3``
  - races: Black, White, Asian, Hispanic
  - genders: male, female
  - annotations: eye centers (provided in the interface)


*  `CMU Multi-PIE face database <http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html>`__  (:any:`bob.bio.face.database.MultipieDatabase`)

  - directory key: ``bob.db.multipie.directory``; the directory containing all the images
  - extension: ``.png``
  - protocols: ``G, E, U, M, P, P240, P191, P130, P010, P041, P051, P050, P110, P140, P200, P190, P120, P080, P081, P090,``
  - annotations: eye centers (provided in the interface)


* PolaThermal database (:any:`bob.bio.face.database.PolaThermalDatabase`)

  - directory key: ``bob.db.pola-thermal.directory``; the directory containing all the images
  - expected extension: ``.png``
  - protocols: There are more than 30 protocols. Here we list the most important ones: ``VIS-thermal-overall-split1, VIS-thermal-overall-split2, VIS-thermal-overall-split3, VIS-thermal-overall-split4, VIS-thermal-overall-split5, "VIS-polarimetric-overall-split1, VIS-polarimetric-overall-split2, VIS-polarimetric-overall-split3, VIS-polarimetric-overall-split4, VIS-polarimetric-overall-split5, ```
  - annotations: eye centers (provided in the interface)


* Racial Faces in the Wild (RFW) (:any:`bob.bio.face.database.RFWDatabase`)

  - directory key: ``bob.bio.face.rfw.directory``; the directory containing all the images
  - expected extension: ``.png``
  - protocols: ``original, idiap``. The idiap protocol is an extension of the original protocol, where it is allowed comparison between samples from all races.
  - annotations: eye centers (provided in the interface)
  - races: African, Asian, Caucasian, Indian
  - note: In this dataset we used the `Wikidata <https://query.wikidata.org/>`__ to extend its metadata by adding the gender and the country information.


* `Surveillance Camera Face Database <https://www.scface.org/>`__  (:any:`bob.bio.face.database.SCFaceDatabase`)

  - directory key: ``bob.bio.face.scface.directory``; the directory containing all the images
  - expected extension:
  - protocols: ``close, medium, far, combined, IR``
  - annotations: eye centers (provided in the interface)
