.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Mon 13 Aug 2012 12:36:40 CEST

.. _bob.bio.face:

===========================================
 Face Recognition Algorithms and Databases
===========================================

This package is part of the ``bob.bio`` packages, which provide open source tools to run comparable and reproducible biometric recognition experiments.
In this package, tools for executing face recognition experiments are provided.
This includes:

* Preprocessors to detect, align and photometrically enhance face images
* Feature extractors that extract features from facial images
* Recognition algorithms that are specialized on facial features, and
* Facial image databases including their protocols.

Additionally, a set of baseline algorithms are defined, which integrate well with the two other ``bob.bio`` packages:

* :ref:`bob.bio.gmm <bob.bio.gmm>` defines algorithms based on Gaussian mixture models
* :ref:`bob.bio.video <bob.bio.video>` uses face recognition algorithms in video frames
* :ref:`bob.bio.csu <bob.bio.csu>` provides wrapper classes of the `CSU Face Recognition Resources <http://www.cs.colostate.edu/facerec>`_ (only Python 2.7 compatible)

For more detailed information about the structure of the ``bob.bio`` packages, please refer to the documentation of :ref:`bob.bio.base <bob.bio.base>`.
Particularly, the installation of this and other ``bob.bio`` packages, please read the :ref:`bob.bio.base.installation`.

In the following, we provide more detailed information about the particularities of this package only.

===========
Users Guide
===========

.. toctree::
   :maxdepth: 2

   baselines
   implementation

================
Reference Manual
================

.. toctree::
   :maxdepth: 2

   implemented


.. include:: references.rst

.. todolist::
