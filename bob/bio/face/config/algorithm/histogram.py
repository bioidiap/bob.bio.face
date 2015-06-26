#!/usr/bin/env python

import bob.bio.face
import bob.math

algorithm = bob.bio.face.algorithm.Histogram(
    distance_function = bob.math.histogram_intersection,
    is_distance_function = False
)
