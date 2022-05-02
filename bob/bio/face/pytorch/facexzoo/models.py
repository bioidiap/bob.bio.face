import pkg_resources

from bob.bio.face.pytorch.facexzoo.backbone_def import BackboneFactory
from bob.extension.download import get_file

def_backbone_conf = pkg_resources.resource_filename(
    "bob.bio.face", "pytorch/facexzoo/backbone_conf.yaml"
)

info = {
    "AttentionNet": [
        "AttentionNet-f4c6f908.pt.tar.gz",
        "49e435d8d9c075a4f613336090eac242",
    ],
    "ResNeSt": [
        "ResNeSt-e8b132d4.pt.tar.gz",
        "51eef17ef7c17d1b22bbc13022393f31",
    ],
    "MobileFaceNet": [
        "MobileFaceNet-ca475a8d.pt.tar.gz",
        "e5fc0ae59d1a290b58a297b37f015e11",
    ],
    "ResNet": ["ResNet-e07e7fa1.pt.tar.gz", "13596dfeeb7f40c4b746ad2f0b271c36"],
    "EfficientNet": [
        "EfficientNet-5aed534e.pt.tar.gz",
        "31c827017fe2029c1ab57371c8e5abf4",
    ],
    "TF-NAS": ["TF-NAS-709d8562.pt.tar.gz", "f96fe2683970140568a17c09fff24fab"],
    "HRNet": ["HRNet-edc4da11.pt.tar.gz", "5ed9920e004af440b623339a7008a758"],
    "ReXNet": ["ReXNet-7c45620c.pt.tar.gz", "b24cf257a25486c52fde5626007b324b"],
    "GhostNet": [
        "GhostNet-5f026295.pt.tar.gz",
        "9edb8327c62b62197ad023f21bd865bc",
    ],
}


class FaceXZooModelFactory:
    def __init__(self, arch, backbone_conf=def_backbone_conf, info=info):
        self.arch = arch
        self.backbone_conf = backbone_conf
        self.info = info

        assert self.arch in self.info.keys()

    def get_model(self):
        return BackboneFactory(self.arch, self.backbone_conf).get_backbone()

    def get_checkpoint_name(self):
        return self.info[self.arch][0]

    def get_facexzoo_file(self):
        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.learn.pytorch/facexzoomodels/{}".format(
                self.info[self.arch][0]
            ),
            "http://www.idiap.ch/software/bob/data/bob/bob.learn.pytorch/facexzoomodels/{}".format(
                self.info[self.arch][0]
            ),
        ]

        return get_file(
            self.info[self.arch][0],
            urls,
            cache_subdir="data/pytorch/{}/".format(self.info[self.arch][0]),
            file_hash=self.info[self.arch][1],
            extract=True,
        )
