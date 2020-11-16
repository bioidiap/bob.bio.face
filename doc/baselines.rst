.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.face.baselines:

=============================
Executing Baseline Algorithms
=============================

.. todo::
   Here we should:   
     - Brief how to run an experiment
     - Point to bob.bio.base for further explanation
     - Show the baselines available
     - Show the databases available


The baselines
-------------

The algorithms present an (incomplete) set of state-of-the-art face recognition algorithms. Here is the list of short-cuts:

* ``eigenface``: The eigenface algorithm as proposed by [TP91]_. It uses the pixels as raw data, and applies a *Principal Component Analysis* (PCA) on it:

  - preprocessor : :py:class:`bob.bio.face.preprocessor.FaceCrop`
  - feature : :py:class:`bob.bio.base.extractor.Linearize`
  - algorithm : :py:class:`bob.bio.base.algorithm.PCA`

* ``lda``: The LDA algorithm applies a *Linear Discriminant Analysis* (LDA), here we use the combined PCA+LDA approach [ZKC98]_:

  - preprocessor : :py:class:`bob.bio.face.preprocessor.FaceCrop`
  - feature : :py:class:`bob.bio.face.extractor.Eigenface`
  - algorithm : :py:class:`bob.bio.base.algorithm.LDA`

* ``gaborgraph``: This method extract grid graphs of Gabor jets from the images, and computes a Gabor phase based similarity [GHW12]_.

  - preprocessor : :py:class:`bob.bio.face.preprocessor.INormLBP`
  - feature : :py:class:`bob.bio.face.extractor.GridGraph`
  - algorithm : :py:class:`bob.bio.face.algorithm.GaborJet`


Further algorithms are available, when the :ref:`bob.bio.gmm <bob.bio.gmm>` package is installed:

* ``gmm``: *Gaussian Mixture Models* (GMM) [MM09]_ are extracted from *Discrete Cosine Transform* (DCT) block features.

  - preprocessor : :py:class:`bob.bio.face.preprocessor.TanTriggs`
  - feature : :py:class:`bob.bio.face.extractor.DCTBlocks`
  - algorithm : :py:class:`bob.bio.gmm.algorithm.GMM`

* ``isv``: As an extension of the GMM algorithm, *Inter-Session Variability* (ISV) modeling [WMM11]_ is used to learn what variations in images are introduced by identity changes and which not.

  - preprocessor : :py:class:`bob.bio.face.preprocessor.TanTriggs`
  - feature : :py:class:`bob.bio.face.extractor.DCTBlocks`
  - algorithm : :py:class:`bob.bio.gmm.algorithm.ISV`

* ``ivector``: Another extension of the GMM algorithm is *Total Variability* (TV) modeling [WM12]_ (aka. I-Vector), which tries to learn a subspace in the GMM super-vector space.

  - preprocessor : :py:class:`bob.bio.face.preprocessor.TanTriggs`
  - feature : :py:class:`bob.bio.face.extractor.DCTBlocks`
  - algorithm : :py:class:`bob.bio.gmm.algorithm.IVector`

.. note::
  The ``ivector`` algorithm needs a lot of training data and fails on small databases such as the `AT&T database`_.

.. _bob.bio.base.baseline_results:

