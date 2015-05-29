#!/usr/bin/env python

import bob.bio.base
import bob.bio.face
import math

# load the face cropping parameters
cropper = bob.bio.base.load_resource("face-crop-eyes", "preprocessor")

extractor = bob.bio.face.extractor.GridGraph(
    # Gabor parameters
    gabor_sigma = math.sqrt(2.) * math.pi,

    # what kind of information to extract
    normalize_gabor_jets = True,

    # setup of the fixed grid
    node_distance = (4, 4),
    first_node = (6, 6),
    image_resolution = cropper.cropped_image_size
)
