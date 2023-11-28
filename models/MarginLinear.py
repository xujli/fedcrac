import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from scipy.special import binom


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)


class LGMLoss(nn.Module):
    """
    Refer to paper:
    Weitao Wan, Yuanyi Zhong,Tianpeng Li, Jiansheng Chen
    Rethinking Feature Distribution for Loss Functions in Image Classification. CVPR 2018
    re-implement by yirong mao
    2018 07/02
    """

    def __init__(self, num_classes, feat_dim, alpha):
        super(LGMLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.log_covs = nn.Parameter(torch.zeros(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        log_covs = torch.unsqueeze(self.log_covs, dim=0)

        covs = torch.exp(log_covs)  # 1*c*d
        tcovs = covs.repeat(batch_size, 1, 1)  # n*c*d
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1)  # eq.(18)

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.alpha)
        y_onehot = y_onehot + 1.0
        margin_dist = torch.mul(dist, y_onehot)

        slog_covs = torch.sum(log_covs, dim=-1)  # 1*c
        tslog_covs = slog_covs.repeat(batch_size, 1)
        margin_logits = -0.5 * (tslog_covs + margin_dist)  # eq.(17)
        logits = -0.5 * (tslog_covs + dist)

        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        cdist = cdiff.pow(2).sum(1).sum(0) / 2.0

        slog_covs = torch.squeeze(slog_covs)
        reg = 0.5 * torch.sum(torch.index_select(slog_covs, dim=0, index=label.long()))
        likelihood = (1.0 / batch_size) * (cdist + reg)

        return logits, margin_logits, likelihood


class LGMLoss_v0(nn.Module):
    """
    LGMLoss whose covariance is fixed as Identity matrix
    """

    def __init__(self, num_classes, feat_dim, alpha):
        super(LGMLoss_v0, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feat, label):
        batch_size = feat.shape[0]

        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        diff = torch.mul(diff, diff)
        dist = torch.sum(diff, dim=-1)

        y_onehot = one_hot(label, self.num_classes) * self.alpha
        y_onehot = y_onehot + 1.0
        margin_dist = torch.mul(dist, y_onehot)
        margin_logits = -0.5 * margin_dist
        logits = -0.5 * dist

        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        likelihood = (1.0 / batch_size) * cdiff.pow(2).sum(1).sum(0) / 2.0
        return logits, margin_logits, likelihood


class CosineSimilarity(nn.Module):
    def __init__(self, in_features, out_features, scale_factor=30.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).float())
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature):
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        return cosine * self.scale_factor


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale_factor: norm of input feature
        margin: margin
    :return： (theta) - m
    """

    def __init__(self, in_features, out_features, scale_factor=30.0, margin=0.40):
        super(AddMarginProduct, self).__init__()
        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))

        # when test, no label, just return
        if label is None:
            return cosine * self.scale_factor

        phi = cosine - self.margin
        output = torch.where(
            one_hot(label, cosine.shape[1]).byte(), phi, cosine)
        output *= self.scale_factor

        return


class SoftmaxMarginL(nn.Module):
    r"""Implement of softmax with margin:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale_factor: norm of input feature
        margin: margin
    """

    def __init__(self, in_features, out_features, scale_factor=5.0, margin=1):
        super(SoftmaxMarginL, self).__init__()
        self.scale_factor = scale_factor
        self.margin = torch.tensor(margin)
        self.coef = nn.Parameter(torch.ones(1))
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        z = F.linear(feature, self.weight)
        z -= z.min(dim=1, keepdim=True)[0]

        # when test, no label, just return
        if label is None:
            return z * self.scale_factor

        if len(self.margin.shape) == 0:
            phi = z - self.coef + self.margin
            output = torch.where(
                one_hot(label, z.shape[1]).byte(), phi, z)
        elif len(self.margin.shape) == 1:
            phi = z - torch.unsqueeze(self.coef + self.margin[label], -1)
            output = torch.where(
                one_hot(label, z.shape[1]).byte(), phi, z)
        elif len(self.margin.shape) == 2:
            phi = z + self.coef + self.margin[label]
            output = torch.where(1 - one_hot(label, z.shape[1]).byte(), phi, z)

        output *= self.scale_factor

        return output


class SoftmaxMargin(nn.Module):
    r"""Implement of softmax with margin:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale_factor: norm of input feature
        margin: margin
    """

    def __init__(self, in_features, out_features, scale_factor=5.0, margin=1):
        super(SoftmaxMargin, self).__init__()
        self.scale_factor = scale_factor
        self.margin = torch.tensor(margin)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        z = F.linear(feature, self.weight)
        z -= z.min(dim=1, keepdim=True)[0]

        # when test, no label, just return
        if label is None:
            return z * self.scale_factor

        if len(self.margin.shape) == 0:
            phi = z - self.margin
            output = torch.where(
                one_hot(label, z.shape[1]).byte(), phi, z)
        elif len(self.margin.shape) == 1:
            phi = z - torch.unsqueeze(self.margin[label], -1)
            output = torch.where(
                one_hot(label, z.shape[1]).byte(), phi, z)
        elif len(self.margin.shape) == 2:
            phi = z + self.margin[label]
            output = torch.where(1 - one_hot(label, z.shape[1]).byte(), phi, z)

        output *= self.scale_factor

        return output


class SoftmaxMarginMix(nn.Module):
    r"""Implement of softmax with margin:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale_factor: norm of input feature
        margin: margin
    """

    def __init__(self, in_features, out_features, scale_factor=5.0, margin=1):
        super(SoftmaxMarginMix, self).__init__()
        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, labels=None):
        z = F.linear(feature, self.weight)
        z -= z.min(dim=1, keepdim=True)[0]
        # when test, no label, just return
        if labels is None:
            return z * self.scale_factor
        # print(z.shape)
        label_a, label_b = labels
        phi = z - self.margin
        output_a = torch.where(
            one_hot(label_a, z.shape[1]).byte(), phi, z)
        output_a *= self.scale_factor

        output_b = torch.where(
            one_hot(label_b, z.shape[1]).byte(), phi, z)
        output_b *= self.scale_factor

        return output_a, output_b


class MARC(nn.Module):
    def __init__(self, in_features, out_features):
        super(MARC, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features), requires_grad=True)
        self.omega = torch.nn.Parameter(data=torch.ones(1, out_features), requires_grad=True)
        self.beta = torch.nn.Parameter(data=torch.zeros(1, out_features), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        z = F.linear(feature, self.weight)
        # when test, no label, just return
        if label is None:
            return z * self.scale_factor
        with torch.no_grad():
            w_norm = torch.norm(self.weight.data, dim=1)
        output = self.omega * z + self.beta * w_norm

        return output


class DisAlignLinear(torch.nn.Linear):
    """
    A wrapper for nn.Linear with support of DisAlign method.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.confidence_layer = torch.nn.Linear(in_features, 1)
        self.logit_scale = torch.nn.Parameter(torch.ones(1, out_features))
        self.logit_bias = torch.nn.Parameter(torch.zeros(1, out_features))
        torch.nn.init.constant_(self.confidence_layer.weight, 0.1)

    def forward(self, input: torch.Tensor, target=None):
        logit_before = F.linear(input, self.weight, self.bias)
        confidence = self.confidence_layer(input).sigmoid()
        logit_after = (1 + confidence * self.logit_scale) * logit_before + \
                      confidence * self.logit_bias
        return logit_after


class LSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin, device):
        super().__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99

        self.device = device  # gpu or cpu

        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).to(device)  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device)  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device)  # n
        self.signs = torch.ones(margin // 2 + 1).to(device)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta ** 2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)
            logit = x.mm(w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm *
                                    x_norm *
                                    (((-1) ** k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

            logit[indexes, target] = logit_target_updated_beta
            self.beta *= self.scale
            return logit
        else:
            assert target is None
            return input.mm(self.weight)
