.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <siebenkopf@googlemail.com>

.. _bob.bio.face.datasets:

Setting up Facial Image Databases
=================================

This package includes a large list of facial image datasets including their default evaluation protocols.
You can list all available datasets and protocols by calling:

.. code-block:: bash

  resources.py --types databases


We use the default bob configuration system for setting up database interfaces, which basically translates to:

.. code-block:: bash

  bob config set <key> <value>

Mainly, we need to tell the system where the original files can be located.
In some circumstances, the filename extension of the datasets can change from version to version, so they can be setup as well.
In the following, find a list of all current databases, their configuration parameters, and their provided protocols, in alphabetic order.

* `ARface dataset <https://www2.ece.ohio-state.edu/~aleix/ARdatabase.html>`_ (:any:`bob.bio.face.database.ARFaceDatabase`)

  - directory key: ``bob.bio.face.database.arface.directory``; the directory containing all the images
  - extension key: ``bob.bio.face.database.arface.extension``; ``".ppm"`` if not set
  - protocols: ``expression, illumination, occlusion, occlusion_and_illumination, all``
  - annotations: eye centers (provided in the interface)
  - notes: The original files are stored in a raw format and need to be converted into some image format.

* AT&T dataset of faces, previously known as the ORL dataset (:any:`bob.bio.face.database.AtntBioDatabase`)

  - annotations: none, faces are already aligned
  - notes: The dataset will be downloaded completely, including all images. No special setup required.

* Casia Africa Database (:any:`bob.bio.face.database.CasiaAfricaDatabase`)

  - directory key: ``bob.db.casia-africa.directory``; the directory containing all the images
  - extension: ``.jpg``
  - protocols: ``ID-V-All-Ep1, ID-V-All-Ep2, ID-V-All-Ep3``,
  - annotations: eye centers (provided in the interface)
  - notes: The data is very noisy (there are no faces in some images), and the original protocols need to be pruned.


* CASPEAL (:any:`bob.bio.face.database.CaspealDatabase`)

  - directory key: ``bob.bio.face.caspeal.directory``; the directory containing all the images
  - extension: ``.png``
  - protocols: ``accessory,aging,background,distance,expression,lighting,``,
  - annotations: eye centers (provided in the interface)  

* CASIA NIR-VIS 2.0 <http://www.cbsr.ia.ac.cn/english/NIR-VIS-2.0-Database.html>_ (:any:`bob.bio.face.database.CBSRNirVis2Database`)

  - directory key: ``bob.db.cbsr-nir-vis-2.directory``; the directory containing all the images
  - extension: ``[".jpg", ".bmp"]``
  - protocols: ``view2_1, view2_2, view2_3, view2_4, view2_5, view2_6, view2_7, view2_8, view2_9, view2_10``
  - annotations: eye centers (provided in the interface) 

* FRGC (:any:`bob.bio.face.database.FRGCDatabase`)

  - directory key: ``bob.bio.face.frgc.directory``; the directory containing all the images
  - extension:
  - protocols: ``2.0.1,2.0.2,2.0.4``
  - annotations: eye centers (provided in the interface) 


* The Good, Bad and Ugly database (GBU) (:any:`bob.bio.face.database.GBUDatabase`)

  - directory key: ``bob.bio.face.gbu.directory``; the directory containing all the images
  - extension: ``.jpg``
  - protocols: ``Good, Bad, Ugly``
  - annotations: eye centers (provided in the interface) 


* Labeled Faced in the Wild <http://vis-www.cs.umass.edu/lfw>`_ (LFW) database (:any:`bob.bio.face.database.LFWDatabase`)

  - directory key: ``bob.bio.face.lfw.directory``; the directory containing all the images
  - extension: ``.jpg``
  - protocols: ``view2``
  - annotations: There are three types of eyes annotations: ``funneled, idiap, named`` (provided in the interface) 


*  MEDS II (:any:`bob.bio.face.database.MEDSDatabase`)

  - directory key: ``bob.db.meds.directory``; the directory containing all the images
  - extension: ``.jpg``
  - protocols: ``verification_fold1, verification_fold2, verification_fold3``
  - races: Black, White
  - genders: male
  - annotations: eye centers (provided in the interface) 

*  MOBIO dataset (:any:`bob.bio.face.database.MobioDatabase`)

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

*  CMU Multi-PIE face database <http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html>`_  (:any:`bob.bio.face.database.MultipieDatabase`)

  - directory key: ``bob.db.multipie.directory``; the directory containing all the images
  - extension: ``.png``
  - protocols: ``P240, P191, P130, G, P010, P041, P051, P050, M, P110, P, P140, U, P200, E, P190, P120, P080, P081, P090,``
  - annotations: eye centers (provided in the interface) 

* PolaThermal database (:any:`bob.bio.face.database.PolaThermalDatabase`)

  - directory key: ``bob.db.pola-thermal.directory``; the directory containing all the images
  - extension: ``.png``
  - protocols: There are more than 30 protocols. Here we list the most important ones: ``VIS-thermal-overall-split1, VIS-thermal-overall-split2, VIS-thermal-overall-split3, VIS-thermal-overall-split4, VIS-thermal-overall-split5, "VIS-polarimetric-overall-split1, VIS-polarimetric-overall-split2, VIS-polarimetric-overall-split3, VIS-polarimetric-overall-split4, VIS-polarimetric-overall-split5, ```
  - annotations: eye centers (provided in the interface) 


* Racial Faces in the Wild (RFW) (:any:`bob.bio.face.database.RFWDatabase`)

  - directory key: ``bob.bio.face.rfw.directory``; the directory containing all the images
  - races: African, Asian, Caucasian, Indian
  - extension: ``.png``
  - protocols: ``original, idiap``. The idiap protocol is an extension of the original protocol, where it is allowed comparison between samples from all races.
  - annotations: eye centers (provided in the interface) 
  - note: In this dataset we used the wikidata <https://query.wikidata.org/>_ to extend its metadata by adding the gender and the country information.


* SCFace (:any:`bob.bio.face.database.SCFaceDatabase`)

  - directory key: ``bob.bio.face.scface.directory``; the directory containing all the images
  - extension: 
  - protocols: ``close, medium, far, combined, IR``
  - annotations: eye centers (provided in the interface)