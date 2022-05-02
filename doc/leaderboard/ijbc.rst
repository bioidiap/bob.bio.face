.. vim: set fileencoding=utf-8 :

.. _bob.bio.face.learderboard.ijbc:

=============
IJB-C Dataset
=============


The IARPA Janus Benchmark C (IJB-C) is one of the most challenging evaluation datasets in face recognition research.
This dataset contains evaluation protocols for face detection, face clustering, face verification, and open-set face identification.
Included in the database, there are list files defining verification as well open-set identification protocols.
For verification, two different protocols are provided (`ijbc-test1` and `ijbc-test2`) so as for the open-set (`ijbc-test4-g1` and `ijbc-test4-g2`).




Setting up the database
=======================

To use this dataset protocol, you need to have the original files of the IJBC datasets.
Once you have it downloaded, please run the following command to set the path for Bob

   .. code-block:: sh

      bob config set bob.bio.face.ijbc.directory [IJBC PATH]


Benchmarking
============

You can run the IJBC baselines command with a simple command such as:

.. code-block:: bash

   bob bio pipeline simple ijbc-test4-g1 iresnet100


The above command will run an example of open set face recognition.


:ref:`bob.bio.face` has some customized plots where the FMR and FNMR trade-off in the evaluation set can be plot using operational
FMR thresholds from the development set.
This is done be the command `bob bio face plots ijbc` command as in the example below:


.. code-block:: bash

   wget https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/scores/frice_scores.tar.gz
   tar -xzvf frice_scores.tar.gz
   bob bio face plots ijbc \
        ./frice_scores/ijbc/arcface_insightface/scores-dev.csv \
        ./frice_scores/ijbc/resnet50_msceleb_arcface_20210521/scores-dev.csv \
        ./frice_scores/ijbc/attention_net/scores-dev.csv \
        ./frice_scores/ijbc/facenet_sanderberg/scores-dev.csv \
        --titles ArcFace-100,Idiap-Resnet50,Zoo-AttentionNet,Facenet-Sandberg -o plot.pdf
