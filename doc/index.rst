.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Mon 13 Aug 2012 12:36:40 CEST

.. _bob.bio.face:

=============================
 Open Source Face Recognition
=============================


This package provide open source tools to run comparable and reproducible face recognition experiments.
This includes:

* Preprocessors to detect, align and photometrically enhance face images
* Feature extractors that extract features from facial images
* Facial image databases including their protocols.
* Scripts that trains CNNs

For more detailed information about how this package is structured, please refer to the documentation of :ref:`bob.bio.base <bob.bio.base>`.


Get Started
============

The easiest way to get started is by simply comparing two faces::

$ bob bio compare-samples -p gabor_graph me.png not_me.png

.. warning::
   No face detection is carried out with this command.

Check out all the face recognition algorithms available by doing::

$ resources.py --type p


Get Started, serious 
====================

.. todo::

   Briefing about baselines
 

Users Guide
===========

.. toctree::
   :maxdepth: 2

   baselines
   leaderboad
   references
   annotators

Reference Manual
================

.. toctree::
   :maxdepth: 2

   implemented


.. todolist::
