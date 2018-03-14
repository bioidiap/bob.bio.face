.. _bob.bio.face.implemented:

=================================
Tools implemented in bob.bio.face
=================================

Summary
-------

Databases
~~~~~~~~~

.. autosummary::
   bob.bio.face.database.ARFaceBioDatabase
   bob.bio.face.database.AtntBioDatabase
   bob.bio.face.database.BancaBioDatabase
   bob.bio.face.database.MobioBioDatabase
   bob.bio.face.database.CaspealBioDatabase
   bob.bio.face.database.ReplayBioDatabase
   bob.bio.face.database.ReplayMobileBioDatabase
   bob.bio.face.database.MsuMfsdModBioDatabase
   bob.bio.face.database.GBUBioDatabase
   bob.bio.face.database.LFWBioDatabase
   bob.bio.face.database.MultipieBioDatabase
   bob.bio.face.database.IJBABioDatabase
   bob.bio.face.database.XM2VTSBioDatabase
   bob.bio.face.database.FRGCBioDatabase
   bob.bio.face.database.SCFaceBioDatabase


Face Image Annotators
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.face.annotator.Base
   bob.bio.face.annotator.BobIpFacedetect
   bob.bio.face.annotator.BobIpFlandmark
   bob.bio.face.annotator.BobIpDlib
   bob.bio.face.annotator.BobIpMTCNN


Image Preprocessors
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.face.preprocessor.Base
   bob.bio.face.preprocessor.FaceCrop
   bob.bio.face.preprocessor.FaceDetect

   bob.bio.face.preprocessor.TanTriggs
   bob.bio.face.preprocessor.HistogramEqualization
   bob.bio.face.preprocessor.SelfQuotientImage
   bob.bio.face.preprocessor.INormLBP


Image Feature Extractors
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.face.extractor.Eigenface
   bob.bio.face.extractor.DCTBlocks
   bob.bio.face.extractor.GridGraph
   bob.bio.face.extractor.LGBPHS


Face Recognition Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.face.algorithm.GaborJet
   bob.bio.face.algorithm.Histogram


Databases
---------
.. automodule:: bob.bio.face.database

Annotators
----------

.. automodule:: bob.bio.face.annotator

Preprocessors
-------------

.. automodule:: bob.bio.face.preprocessor

Extractors
----------

.. automodule:: bob.bio.face.extractor

Algorithms
----------

.. automodule:: bob.bio.face.algorithm


.. include:: links.rst
