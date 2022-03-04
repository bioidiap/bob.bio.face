.. _bob.bio.face.faq:

===
FAQ
===

How to crop a face ?
====================

The recipe below helps you to set a face cropper based on eye positions.

.. literalinclude:: faq/facecrop.py

.. figure:: ./faq/img/ada_cropped.png


How to choose the cropped positions ?
=====================================

The ideal cropped positions are dependent on the specific application you are using the face cropper in.
Some face embedding extractors work well on loosely cropped faces, while others require the face to be tightly cropped.
We provide a few reasonable defaults that are used in our implemented baselines. They are accessible through a function as follows :
::

    from bob.bio.face.utils import get_default_cropped_positions
    mode = 'legacy'
    cropped_image_size=(160, 160)
    annotation_type='eyes-center'
    cropped_positions = get_default_cropped_positions(mode, cropped_image_size, annotation_type)


There are currently three available modes :

* :code:`legacy` Tight crop, used in non neural-net baselines such as :code:`gabor-graph`, :code:`lgbphs`.
  It is typically use with a 5:4 aspect ratio for the :code:`cropped_image_size`
* :code:`dnn` Loose crop, used for neural-net baselines such as the ArcFace or FaceNet models.
* :code:`pad` Tight crop used in some PAD baselines

We present hereafter a visual example of those crops for the `eyes-center` annotation type.

.. plot:: plot/default_crops.py
    :include-source: True
