#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import torchvision.transforms as transforms

import bob.io.image


def get_standard_data_augmentation():
    """
    Standard data augmentation used on my experiments
    """
    transform = transforms.Compose(
        [
            lambda x: bob.io.image.to_matplotlib(x),
            lambda x: x.astype("uint8"),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-3, 3)),
            transforms.RandomAutocontrast(p=0.1),
            transforms.PILToTensor(),
            lambda x: (x - 127.5) / 128.0,
        ]
    )
    return transform
