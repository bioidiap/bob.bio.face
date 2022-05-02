from torch.nn import Linear, Module


class Regular(Module):
    """
    Implement a regular head used for softmax layers
    """

    def __init__(self, feat_dim, num_class):
        super(Regular, self).__init__()

        self.fc = Linear(feat_dim, num_class, bias=False)

    def forward(self, feats, labels):
        return self.fc(feats)
