import numpy

import bob.bio.face.annotator as fd


def test_bbx():
    # Tests that the bounding box calculation works as expected

    # check the indirect ways using eye coordinates
    bb = fd.bounding_box_from_annotation(
        leye=(10, 10),
        reye=(10, 30),
        padding={"left": -1, "right": 1, "top": -1, "bottom": 1},
    )
    assert bb.topleft == (-10, 0)
    assert bb.bottomright == (30, 40)
    assert bb.size == (40, 40)

    assert bb.contains((-10, 0))
    assert bb.contains((29, 39))
    assert not bb.contains((100, 100))
    assert not bb.contains((-100, -100))

    # check the scaling functionality
    sbb = bb.scale(0.5)
    assert sbb.topleft == (-5, 0)
    assert sbb.bottomright == (15, 20)
    sbb = bb.scale(2.0, centered=True)
    assert sbb.topleft == (-30, -20)
    assert sbb.bottomright == (50, 60)
    sbb = bb.scale(0.84)
    assert sbb.topleft == (-8, 0)
    assert sbb.bottomright == (25, 34)

    bb = fd.bounding_box_from_annotation(leye=(10, 10), reye=(10, 30))
    assert bb.topleft == (-4, 0)
    assert bb.bottomright == (44, 40)

    # test that left and right profile versions work
    lbb = fd.bounding_box_from_annotation(
        source="left-profile", mouth=(40, 10), eye=(20, 10)
    )
    assert lbb.topleft == (10, 6)
    assert lbb.bottomright == (50, 26)

    # test the direct way
    bb1 = fd.bounding_box_from_annotation(
        topleft=(10, 20), bottomright=(30, 40)
    )
    assert bb1.topleft == (10, 20)
    assert bb1.bottomright == (30, 40)
    assert bb1.area == 400

    bb2 = fd.bounding_box_from_annotation(
        topleft=(15, 25), bottomright=(35, 45)
    )
    bb3 = bb1.overlap(bb2)
    assert bb3.topleft == (15, 25)
    assert bb3.bottomright == (30, 40)
    assert bb3.area == 225

    # check the similarity function
    assert bb1.similarity(bb1) == 1.0
    assert bb3.similarity(bb2) == 0.5625
    assert bb3.similarity(bb1) == bb1.similarity(bb3)


def test_mirror():
    bb = fd.bounding_box_from_annotation(topleft=(10, 20), bottomright=(30, 40))
    mirrored = bb.mirror_x(60)
    assert mirrored.top == bb.top
    assert mirrored.bottom == bb.bottom
    assert mirrored.topleft[1], 20
    assert mirrored.bottomright[1], 40

    # test that this IS actually, what we want
    image = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    mirrored_image = image[:, ::-1]
    bb = fd.BoundingBox((1, 1), (2, 2))
    mb = bb.mirror_x(image.shape[1])
    x = image[
        bb.topleft[0] : bb.bottomright[0], bb.topleft[1] : bb.bottomright[1]
    ]
    y = mirrored_image[
        mb.topleft[0] : mb.bottomright[0], mb.topleft[1] : mb.bottomright[1]
    ]
    assert (x == y[:, ::-1]).all()
