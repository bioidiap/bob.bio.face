.. vim: set fileencoding=utf-8 :
.. Sat Aug 20 07:33:55 CEST 2016

.. image:: https://img.shields.io/badge/docs-v7.0.0-orange.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.bio.face/v7.0.0/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.bio.face/badges/v7.0.0/pipeline.svg
   :target: https://gitlab.idiap.ch/bob/bob.bio.face/commits/v7.0.0
.. image:: https://gitlab.idiap.ch/bob/bob.bio.face/badges/v7.0.0/coverage.svg
   :target: https://gitlab.idiap.ch/bob/bob.bio.face/commits/v7.0.0
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.bio.face


================================
 Run face recognition algorithms
================================

This package is part of the signal-processing and machine learning toolbox
Bob_.
This package is part of the ``bob.bio`` packages, which allow to run comparable and reproducible biometric recognition experiments on publicly available databases.

This package contains functionality to run face recognition experiments.
It is an extension to the `bob.bio.base <http://pypi.python.org/pypi/bob.bio.base>`_ package, which provides the basic scripts.
In this package, utilities that are specific for face recognition are contained, such as:

* Image databases
* Image preprocesors, including face detection and facial image alignment
* Image feature extractors
* Recognition algorithms based on image features



Installation
------------

Complete Bob's `installation`_ instructions. Then, to install this package,
run::

  $ conda install bob.bio.face


Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://www.idiap.ch/software/bob/install
.. _mailing list: https://www.idiap.ch/software/bob/discuss
