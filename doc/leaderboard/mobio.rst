.. vim: set fileencoding=utf-8 :

.. _bob.bio.face.learderboard.mobio:

=============
Mobio Dataset
=============


The MOBIO dataset is a video database containing bimodal data (face/speaker).
It is composed by 152 people (split in the two genders male and female), mostly Europeans, split in 5 sessions (few weeks time lapse between sessions).
The database was recorded using two types of mobile devices: mobile phones (NOKIA N93i) and laptop 
computers(standard 2008 MacBook).

For face recognition images are used instead of videos.
One image was extracted from each video by choosing the video frame after 10 seconds.
The eye positions were manually labelled and distributed with the database.

For more information check:

.. code-block:: latex

    @article{McCool_IET_BMT_2013,
        title = {Session variability modelling for face authentication},
        author = {McCool, Chris and Wallace, Roy and McLaren, Mitchell and El Shafey, Laurent and Marcel, S{\'{e}}bastien},
        month = sep,
        journal = {IET Biometrics},
        volume = {2},
        number = {3},
        year = {2013},
        pages = {117-129},
        issn = {2047-4938},
        doi = {10.1049/iet-bmt.2012.0059},
    }


Benchmarks
==========
    
You can run the mobio baselines command with a simple command such as:

.. code-block:: bash

   bob bio pipeline vanilla-biometrics mobio-male arcface-insightface


Scores from some of our baselines can be found `here <https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/scores/mobio-male.tar.gz>`_.
A det curve can be generated with these scores by running the following commands:

.. code-block:: bash

   wget https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/scores/mobio-male.tar.gz   
   tar -xzvf mobio-male.tar.gz
   bob bio det ./mobio-male/{arcface_insightFace_lresnet100,inception_resnet_v2_msceleb_centerloss_2018,iresnet50,iresnet100,mobilenetv2_msceleb_arcface_2021,resnet50_msceleb_arcface_20210521,vgg16_oxford_baseline,afffe_baseline}/scores-{dev,eval} --legends arcface_insightFace_lresnet100,inception_resnet_v2_msceleb_centerloss_2018,iresnet50,iresnet100,mobilenetv2_msceleb_arcface_2021,resnet50_msceleb_arcface_20210521,vgg16_oxford_baseline,afffe -S -e --figsize 16,8

and get the following :download:`plot <./plots/det-mobio-male.pdf>`.


