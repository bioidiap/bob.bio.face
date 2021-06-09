.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Mon 13 Aug 2012 12:36:40 CEST

.. _bob.bio.face:

=====================================
 Open Source Face Recognition Library
=====================================


This package provide open source tools to run comparable and reproducible face recognition experiments.
This includes:

* Preprocessors to detect, align and photometrically enhance face images
* Feature extractors that extract features from facial images
* Facial image databases including their protocols.
* Scripts that trains CNNs for face recognition.


Get Started
===========

The easiest way to get started is by simply comparing two faces::

$ bob bio compare-samples -p facenet-sanderberg me.png not_me.png

.. warning::
   No face detection is carried out with this command.

Check out all the face recognition algorithms available by doing::

$ resources.py --types p


Get Started, serious 
====================

For detailed information on how this package is structured and how
to run experiments with it, please refer to the documentation of :ref:`bob.bio.base <bob.bio.base>`
and get to know the vanilla biometrics and how to integrate both, algorithm and database protocols with it.
 

Users Guide
===========

.. toctree::
   :maxdepth: 2

   baselines
   leaderboard/leaderboard   
   references
   annotators
   faq

Reference Manual
================

.. toctree::
   :maxdepth: 2

   implemented


.. todolist::
