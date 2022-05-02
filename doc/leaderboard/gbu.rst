.. vim: set fileencoding=utf-8 :

.. _bob.bio.face.learderboard.gbu:

===========
GBU Dataset
===========

The GBU (Good, Bad and Ugly) database consists of parts of the MBGC-V1 image set.
It defines three protocols, i.e., `Good`, `Bad` and `Ugly` for which different model and probe images are used.



Setting up the database
=======================


To use this dataset protocol, you need to have the original files of the IJBC datasets.
Once you have it downloaded, please run the following command to set the path for Bob

   .. code-block:: sh

      bob config set bob.bio.face.gbu.directory [GBU PATH]


Benchmarking
============

You can run the mobio baselines command with a simple command such as:

.. code-block:: bash

   bob bio pipeline simple gbu-good iresnet100


:ref:`bob.bio.face` has some customized plots where the FMR and FNMR trade-off in the evaluation set can be plot using operational
FMR thresholds from the development set.
This is done be the command `bob bio face plots gbu` command as in the example below:


.. code-block:: bash

   wget https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/scores/frice_scores.tar.gz
   tar -xzvf frice_scores.tar.gz
   bob bio face plots gbu \
        ./frice_scores/gbu/good/arcface_insightface/scores-dev.csv \
        ./frice_scores/gbu/good/iresnet50_msceleb_arcface_20210623/scores-dev.csv \
        ./frice_scores/gbu/good/attention_net/scores-dev.csv \
        ./frice_scores/gbu/good/facenet_sanderberg/scores-dev.csv \
        ./frice_scores/gbu/good/ISV/scores-dev.csv \
        --titles ArcFace-100,Idiap-Resnet50,Zoo-AttentionNet,Facenet-Sandberg,2014-ISV -o plot_good.pdf
   bob bio face plots gbu \
        ./frice_scores/gbu/bad/arcface_insightface/scores-dev.csv \
        ./frice_scores/gbu/bad/iresnet50_msceleb_arcface_20210623/scores-dev.csv \
        ./frice_scores/gbu/bad/attention_net/scores-dev.csv \
        ./frice_scores/gbu/bad/facenet_sanderberg/scores-dev.csv \
        ./frice_scores/gbu/bad/ISV/scores-dev.csv \
        --titles ArcFace-100,Idiap-Resnet50,Zoo-AttentionNet,Facenet-Sandberg,2014-ISV -o plot_bad.pdf
   bob bio face plots gbu \
        ./frice_scores/gbu/ugly/arcface_insightface/scores-dev.csv \
        ./frice_scores/gbu/ugly/iresnet50_msceleb_arcface_20210623/scores-dev.csv \
        ./frice_scores/gbu/ugly/attention_net/scores-dev.csv \
        ./frice_scores/gbu/ugly/facenet_sanderberg/scores-dev.csv \
        ./frice_scores/gbu/ugly/ISV/scores-dev.csv \
        --titles ArcFace-100,Idiap-Resnet50,Zoo-AttentionNet,Facenet-Sandberg,2014-ISV -o plot_ugly.pdf

.. note::
  Always remember, `bob bio face plots --help` is your friend.
