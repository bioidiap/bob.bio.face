.. vim: set fileencoding=utf-8 :

.. _bob.bio.face.learderboard.arface:

==============
ARFACE Dataset
==============

Our version of the AR face database contains 3312 images from 136 persons, 76 men and 60 women.
We split the database into several protocols that we have designed ourselves.
The identities are split up into three groups:

* the 'world' group for training your algorithm
* the 'dev' group to optimize your algorithm parameters on
* the 'eval' group that should only be used to report results

Additionally, there are different protocols:

* ``'expression'``: only the probe files with different facial expressions are selected
* ``'illumination'``: only the probe files with different illuminations are selected
* ``'occlusion'``: only the probe files with normal illumination and different accessories (scarf, sunglasses) are selected
* ``'occlusion_and_illumination'``: only the probe files with strong illumination and different accessories (scarf, sunglasses) are selected
* ``'all'``: all files are used as probe

In any case, the images with neutral facial expression, neutral illumination and without accessories are used for enrollment.


Setting up the database
=======================


To use this dataset protocol, you need to have the original files of the Mobio dataset.
Once you have it downloaded, please run the following command to set the path for Bob

   .. code-block:: sh

      bob config set bob.bio.face.arface.directory [ARFACE PATH]



Benchmarking
============

You can run the arface baselines command with a simple command such as:

.. code-block:: bash

   bob bio pipeline simple arface-all iresnet100


:ref:`bob.bio.face` has some customized plots where the FMR and FNMR trade-off in the evaluation set can be plot using operational
FMR thresholds from the development set.
This is done be the command `bob bio face plots arface` command as in the example below:


.. code-block:: bash

   wget https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/scores/frice_scores.tar.gz
   tar -xzvf frice_scores.tar.gz
   bob bio face plots arface -e \
        ./frice_scores/arface/arcface_insightface/scores-{dev,eval}.csv \
        ./frice_scores/arface/iresnet50_msceleb_arcface_20210623/scores-{dev,eval}.csv \
        ./frice_scores/arface/attention_net/scores-{dev,eval}.csv \
        ./frice_scores/arface/facenet_sanderberg/scores-{dev,eval}.csv \
        ./frice_scores/arface/ISV/scores-{dev,eval}.csv \
        --titles ArcFace-100,Idiap-Resnet50,Zoo-AttentionNet,Facenet-Sandberg,2014-ISV -o plot.pdf


.. note::
  Always remember, `bob bio face plots --help` is your friend.
