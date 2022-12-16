import math

import torch
import torch.nn.functional as F

from torch.nn import Module, Parameter


class ArcFace(Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" """

    def __init__(
        self, feat_dim, num_class, margin_arc=0.35, margin_am=0.0, scale=32
    ):
        super(ArcFace, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin_arc = margin_arc
        self.margin_am = margin_am
        self.scale = scale
        self.cos_margin = math.cos(margin_arc)
        self.sin_margin = math.sin(margin_arc)
        self.min_cos_theta = math.cos(math.pi - margin_arc)

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        cos_theta_m = torch.where(
            cos_theta > self.min_cos_theta,
            cos_theta_m,
            cos_theta - self.margin_am,
        )
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output


class MagFace(Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"

    taken from https://github.com/JDAI-CV/FaceX-Zoo/blob/5b63794ba7649fe78a29d2ce0d0216c7773f6174/head/MagFace.py
    """

    def __init__(
        self,
        feat_dim,
        num_class,
        margin_am=0.0,
        scale=32,
        l_a=10,
        u_a=110,
        l_margin=0.45,
        u_margin=0.8,
        lamda=20,
    ):
        super(MagFace, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin_am = margin_am
        self.scale = scale
        self.l_a = l_a
        self.u_a = u_a
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.lamda = lamda

    def calc_margin(self, x):
        margin = (self.u_margin - self.l_margin) / (self.u_a - self.l_a) * (
            x - self.l_a
        ) + self.l_margin
        return margin

    def forward(self, feats, labels):
        x_norm = torch.norm(feats, dim=1, keepdim=True).clamp(
            self.l_a, self.u_a
        )  # l2 norm
        ada_margin = self.calc_margin(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)
        loss_g = 1 / (self.u_a**2) * x_norm + 1 / (x_norm)
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        min_cos_theta = torch.cos(math.pi - ada_margin)
        cos_theta_m = torch.where(
            cos_theta > min_cos_theta, cos_theta_m, cos_theta - self.margin_am
        )
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output, self.lamda * loss_g
