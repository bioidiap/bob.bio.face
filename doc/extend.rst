.. vim: set fileencoding=utf-8 :
.. author: Yu Linghu & Xinyi Zhang <yu.linghu@uzh.ch, xinyi.zhang@uzh.ch>

.. _bob.bio.face.extend:

==================================
Using your own pre-trained network
==================================

While bob includes a plethora of pre-trained face recognition networks, sometimes you want to use a different one and compare it against one of bob's baselines.
For this purpose, we provide several ways to incorporate your own model into bob.
For some frameworks, particularly the most common ones PyTorch, Tensorflow and MxNet, we include specialized interfaces that will allow to use GPU acceleration to extract features.
Additionally, we provide a CPU-only generic function based on OpenCV's ``dnn`` deep learning interface that can handle various models from various frameworks, including the `Open Neural Network Exchange (ONNX) <https://onnx.ai/>`__ format.
In case your framework is not supported, you might want to use `MMDNN <https://github.com/microsoft/MMdnn>`__ to convert your network into one of the accepted formats, preferably ONNX.


Specialized Interfaces
----------------------

As mentioned above, there are three different specialized interfaces that can be used to connect to your network.
All of them have a similar interface, they require a model config and weights (:any:`TensorflowTransformer` only requires one file).
Additionally, each network requires its own image preprocessing, dependent on how on was trained.
For this, you can provide a ``preprocessor``, which is a function taking ``numpy.ndarray`` inputs and returns the preprocessed image.

.. code-block:: python

  transformer = MxNetTransformer(path, weights, preprocessor=lambda x:x/255.)

.. note::
  Images loaded in the Bob echosystem always contains pixel values in range ``[0, 255]``.
  If your model requires pixel values in range ``[-1,1]``, you need to provide a different preprocessor, such as: ``lambda x: (x-127.5)/127.5``.
  Similarly, you can transform RGB images (input) to BGR images (required by the model) by: ``lambda x: x[...,::-1,:,:]``.

Instead of providing the files to load the pre-trained models from, you can always also provide the pre-loaded ``model``.

.. note::
  Since not all of the models are ``pickle``able, though, you might have difficulties with parallelization of the experiments when using these models.

Several models also provide the possibility to assign GPUs for the extraction of features.
By default, features are extracted in CPU mode only.
If you have a GPU available, you can specify these parameters:

* :any:`bob.bio.face.embedding.pytorch.PyTorchModel`: you can directly specify the device you want to use, such as ``device="cuda:3"`` for the GPU with logical index 3.
* :any:`bob.bio.face.embedding.mxnet.MxNetTransformer`: you can specify the logical index of the GPU via ``gpu_id=3``.

.. note::
  When running the pipeline in parallel, each process will use its own copy of the network on the same GPU.
  It is advisable to have few parallel processes assigned, otherwise the GPU might quickly run out of memory.


A Word on Image Alignment
-------------------------

Each pre-trained network will require its own image resolution and type of face alignment, i.e., how much of the face and of the background is in the images.
Unfortunately, information on how to align faces is rarely as precisely given as, e.g., in [GRB17]_.
Thus, you might have some tough time to figure out (approximate) alignment information, i.e., where the eyes are placed in the aligned images.

Most of the datasets provide hand-labeled landmark locations that can be used fort alignment.
Typically, the :any:`bob.bio.face.preprocessor.FaceCrop` will be use to align the face based on the left and right eye locations (in subject perspective, i.e., the ``leye_x`` is usually larger than the ``reye_x``).
These positions of the eyes (in target aligned image coordinates) shall be given as a dictionary: ``{"reye" = (reye_y, reye_x), "leye" = (leye_y, leye_x)}``.

Some datasets do not come with landmark locations, but faces need to be detected.
In order to use a facial landmark :ref:`bob.bio.face.annotator` to detect the landmarks, you can provide an ``annotator``.
This annotator is called for each image and the largest detected face is used and the detected eye landmarks are used for face alignment.
If no face is detected, this face is skipped during scoring.

Finally, some datasets only provide rough bounding box annotations of the faces.
Additionally to the annotator, you need to specify 4 different "landmarks", which are: ``{"reye" = (reye_y, reye_x), "leye" = (leye_y, leye_x), "topleft" = (top, left), "bottomright" = (bottom, right)}``.
When the annotator succeeds, the alignment is done using the annotated eye locations, as above.
When it fails, the face is cropped such that the original bounding box is placed at the given image coordinates.
This functionality is particularly useful for datasets in which images can contain several faces, and the face of interest is marked with a bounding box, such as provided in IJB-C.


Two Complete Examples
---------------------

Importing pre-trained model from library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have found a pretrained model inside of a pytorch package, which we want to test in the pipeline.
For example, we want to use a model from `facenet_pytorch <https://github.com/timesler/facenet-pytorch>`__.
We first need to figure out how to align the face and how to process the images, and we find the parameters to be:

* image resolution: 224x224 pixels
* positions of the eyes in the image: {"leye": (110, 144), "reye": (110, 80)}
* RGB color images
* pixel values are scaled between -1 and 1

Second, we figure out how to load the model ourselves via ``facenet_pytorch.InceptionResnetV1(pretrained="vggface2").eval()``.
Since we can use the landmarks of the dataset, no annotator is required.
That is is, we have everything together that we need.
Now we can create our configuration file and call it ``facenet_pytorch.py``:

.. literalinclude:: examples/facenet_pytorch.py
   :linenos:


Finally, we can run an experiment using the hand-labeled images from the SCface dataset, after you have set up the dataset itself, via:

.. code-block:: bash

  bob bio pipelines vanilla-biometrics scface facenet_pytorch.py


Applying a self-trained tensorflow network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have trained our own model using tensorflow and now want to test how well it performs on the IJB-C dataset.
The model checkpoint is stored in directory ``models/MyTrainedModel``.

.. todo::
  Describe exactly how this directory looks like and what we expect/require to be included.

Since we have trained the model ourselves, we exactly know how we have performed image alignment:

* image resolution: 112x112 pixels
* we have a very tight box of the face, our positions of the eyes in the image are: {"leye": (20, 88), "reye": (20, 24)}
* approximate location of the bounding box; note that target coordinates can be outside of of aligned image coordinates: {"topleft": (-10, 0), "bottomright": (102, 112)}
* RGB color images
* image preprocessing was using the default ImageNet parameters: input pixels between 0 and 1, subtract mean ``[0.485, 0.456, 0.406]`` and divide by standard deviation ``[0.229, 0.224, 0.225]``.

Let's write our configuration file:

.. literalinclude:: examples/mymodel_tensorflow.py
   :linenos:


Finally, we can run an experiment using the open-set face recognition protocols defined on IJB-C, after you have set up the dataset itself, via:

.. code-block:: bash

  bob bio pipelines vanilla-biometrics ijbc-test4-g1 mymodel_tensorflow.py
