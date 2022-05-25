.. _bob.bio.face.implemented:

=================================
Tools implemented in bob.bio.face
=================================

Summary
-------

Databases
~~~~~~~~~


Bob databases (used for biometric testing)
==========================================

.. autosummary::
   bob.bio.face.database.ARFaceDatabase
   bob.bio.face.database.CasiaAfricaDatabase
   bob.bio.face.database.MobioDatabase
   bob.bio.face.database.IJBCDatabase
   bob.bio.face.database.ReplayMobileBioDatabase
   bob.bio.face.database.GBUDatabase
   bob.bio.face.database.LFWDatabase
   bob.bio.face.database.MultipieDatabase
   bob.bio.face.database.MEDSDatabase
   bob.bio.face.database.MorphDatabase
   bob.bio.face.database.PolaThermalDatabase
   bob.bio.face.database.CBSRNirVis2Database
   bob.bio.face.database.SCFaceDatabase
   bob.bio.face.database.CaspealDatabase

Pytorch databases (used on pytorch)
===================================

.. autosummary::
   bob.bio.face.pytorch.datasets.WebFace42M
   bob.bio.face.pytorch.datasets.MedsTorchDataset
   bob.bio.face.pytorch.datasets.MorphTorchDataset
   bob.bio.face.pytorch.datasets.MobioTorchDataset
   bob.bio.face.pytorch.datasets.MSCelebTorchDataset
   bob.bio.face.pytorch.datasets.VGG2TorchDataset



Deep Learning Extractors
~~~~~~~~~~~~~~~~~~~~~~~~


PyTorch models
==============

.. autosummary::

   bob.bio.face.embeddings.pytorch.afffe_baseline
   bob.bio.face.embeddings.pytorch.iresnet34
   bob.bio.face.embeddings.pytorch.iresnet50
   bob.bio.face.embeddings.pytorch.iresnet100
   bob.bio.face.embeddings.pytorch.GhostNet
   bob.bio.face.embeddings.pytorch.ReXNet
   bob.bio.face.embeddings.pytorch.HRNet
   bob.bio.face.embeddings.pytorch.TF_NAS
   bob.bio.face.embeddings.pytorch.ResNet
   bob.bio.face.embeddings.pytorch.EfficientNet
   bob.bio.face.embeddings.pytorch.MobileFaceNet
   bob.bio.face.embeddings.pytorch.ResNeSt
   bob.bio.face.embeddings.pytorch.AttentionNet
   bob.bio.face.embeddings.pytorch.RunnableModel


Tensorflow models
=================

.. autosummary::
   bob.bio.face.embeddings.tensorflow.facenet_sanderberg_20170512_110547
   bob.bio.face.embeddings.tensorflow.resnet50_msceleb_arcface_2021
   bob.bio.face.embeddings.tensorflow.resnet50_msceleb_arcface_20210521
   bob.bio.face.embeddings.tensorflow.resnet50_vgg2_arcface_2021
   bob.bio.face.embeddings.tensorflow.mobilenetv2_msceleb_arcface_2021
   bob.bio.face.embeddings.tensorflow.inception_resnet_v1_msceleb_centerloss_2018
   bob.bio.face.embeddings.tensorflow.inception_resnet_v2_msceleb_centerloss_2018
   bob.bio.face.embeddings.tensorflow.inception_resnet_v1_casia_centerloss_2018
   bob.bio.face.embeddings.tensorflow.inception_resnet_v2_casia_centerloss_2018



MxNET models
============

.. autosummary::
   bob.bio.face.embeddings.mxnet.arcface_insightFace_lresnet100

Caffe models
============

.. autosummary::
   bob.bio.face.embeddings.opencv.vgg16_oxford_baseline




Face Image Annotators
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.face.annotator.Base
   bob.bio.face.annotator.MTCNN
   bob.bio.face.annotator.TinyFace
   bob.bio.face.annotator.FaceXDetector
   bob.bio.face.annotator.FaceX106Landmarks

Annotation Tools
~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.face.annotator.BoundingBox
   bob.bio.face.annotator.bounding_box_from_annotation
   bob.bio.face.annotator.bounding_box_to_annotations
   bob.bio.face.annotator.expected_eye_positions
   bob.bio.face.annotator.min_face_size_validator

Image Preprocessors
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.face.preprocessor.Base
   bob.bio.face.preprocessor.FaceCrop
   bob.bio.face.preprocessor.MultiFaceCrop
   bob.bio.face.preprocessor.BoundingBoxAnnotatorCrop

   bob.bio.face.preprocessor.TanTriggs
   bob.bio.face.preprocessor.HistogramEqualization
   bob.bio.face.preprocessor.INormLBP


Databases
---------
.. automodule:: bob.bio.face.database

Annotators
----------

.. automodule:: bob.bio.face.annotator

Preprocessors
-------------

.. automodule:: bob.bio.face.preprocessor

Utilities
---------

.. automodule:: bob.bio.face.utils

.. include:: links.rst
