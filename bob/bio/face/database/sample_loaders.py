#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""  Sample and Metatada loaders"""


def eyes_annotations_loader(row, header=None):
    """
    Convert  leye_x, leye_y, reye_x, reye_y attributes to `annotations = (leye, reye)`
    """

    def find_attribute(attribute):
        for i, a in enumerate(header):
            if a == attribute:
                return i
        else:
            ValueError(f"Attribute not found in the dataset: {a}")

    eyes = {
        "leye": (row[find_attribute("leye_x")], row[find_attribute("leye_y")]),
        "reye": (row[find_attribute("reye_x")], row[find_attribute("reye_y")]),
    }

    annotation = {"annotations": eyes}

    return annotation
