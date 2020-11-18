.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.face.baselines:

=============================
Executing Baseline Algorithms
=============================


In this section we introduce the baselines available in this pakcage.
To execute one of then in the databases available just run the following command::

$ bob bio pipelines vanilla-biometrics [DATABASE_NAME] [BASELINE]

.. note::
  Both, `[DATABASE_NAME]` and `[BASELINE]` can be either python resources or
  python files.

  Please, refer to :ref:`bob.bio.base <bob.bio.base>` for more information.  



Baselines available
-------------------

The algorithms below constains all the face recognition baselines available.
It is split in two groups, before and after deep learning era.


Before Deep learning era
========================


* ``eigenface``: The eigenface algorithm as proposed by [TP91]_. It uses the pixels as raw data, and applies a *Principal Component Analysis* (PCA) on it.

* ``lda``: The LDA algorithm applies a *Linear Discriminant Analysis* (LDA), here we use the combined PCA+LDA approach [ZKC98]_

* ``gabor_graph``: This method extract grid graphs of Gabor jets from the images, and computes a Gabor phase based similarity [GHW12]_.

* ``lgbphs``: Local Gabor binary pattern histogram sequence (LGBPHS) implemented in [ZSG05]_


Deep learning baselines
=======================

* ``facenet-sanderberg``: FaceNet trained by `David Sanderberg <https://github.com/davidsandberg/facenet>`_

* ``inception-resnetv2-msceleb``: Inception Resnet v2 model trained using the MSCeleb dataset in the context of the work published by [TFP18]_

* ``inception-resnetv1-msceleb``: Inception Resnet v1 model trained using the MSCeleb dataset in the context of the work published by [TFP18]_

* ``inception-resnetv2-casiawebface``: Inception Resnet v2 model trained using the Casia Web dataset in the context of the work published by [TFP18]_

* ``inception-resnetv1-casiawebface``: Inception Resnet v1 model trained using the Casia Web dataset in the context of the work published by [TFP18]_

* ``arcface-insightface``: Arcface model from `Insightface <https://github.com/deepinsight/insightface>`_
