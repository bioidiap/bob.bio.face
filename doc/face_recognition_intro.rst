..  _bob.bio.base.face_recognition_intro:

################
Face Recognition
################

This section will introduce the basics of face recognition (FR).
Since face recognition is a branch under the wing of biometrics, this section assumes that you have at least a basic knowledge of biometric recognition (see :ref:`this<bob.bio.base.biometrics_introduction>` otherwise).

A face recognition system will follow the same principles as a biometric system, that is:

- a sensor (camera) captures the data
- a transformer extracts feature vectors representing the identity of the person. This can be a series of transformers (e.g. preprocessor and extractor).
- a biometric algorithm decides if two extracted feature vectors represent the same identity.


.. todo::
    more

In :ref:`baselines<baselines>`, you can find the combination of different blocks that create common pipelines.

.. include:: links.rst

