import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class CrossEntropy(nn.Module):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()
        self.para_dict = para_dict
        self.num_classes = self.para_dict['num_classes']
        self.num_class_list = self.para_dict['num_class_list']
        self.device = self.para_dict['device']

        self.weight_list = None
        #setting about defferred re-balancing by re-weighting (DRW)
        self.drw = self.para_dict['if_drw']
        self.drw_start_epoch = self.para_dict['drw_start_round']


    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        loss = F.cross_entropy(inputs, targets, weight=self.weight_list)
        return loss

    def update(self, epoch):
        """
        Adopt cost-sensitive cross-entropy as the default
        Args:
            epoch: int. starting from 1.
        """
        start = (epoch-1) // self.drw_start_epoch
        if start and self.drw:
            self.weight_list = torch.FloatTensor(np.array([min(self.num_class_list) / N for N in self.num_class_list])).to(self.device)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, )    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_ep=0):
        super().__init__()

        self.cls_num_list = [i if i != 0 else 0.1 for i in cls_num_list]
        m_list = 1.0 / np.sqrt(np.sqrt(self.cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
        self.m_list = m_list
        self.s = s
        self.reweight_ep = reweight_ep
        self.per_cls_weights = None

    def forward(self, x, target, round_idx=0, device='0'):

        idx = int(round_idx // self.reweight_ep)

        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)

        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        self.m_list = self.m_list.to(device)
        index_float = index.type(torch.FloatTensor).to(device)
        self.per_cls_weights = self.per_cls_weights.to(device)

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - self.s * batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.per_cls_weights)


class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """
    def __init__(
            self,
            bins=30,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        edges = self.edges
        mmt = self.momentum
        self.edges = self.edges.to(target.get_device())
        self.acc_sum = self.acc_sum.to(target.get_device())

        N = pred.size(0)
        C = pred.size(1)
        P = F.softmax(pred)

        class_mask = pred.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)
        g = (g * class_mask).sum(1).view(-1, 1)
        weights = torch.zeros_like(g)

        # valid = label_weight > 0
        # tot = max(valid.float().sum().item(), 1.0)
        tot = pred.size(0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) 
            num_in_bin = inds.sum().item()
            # print(num_in_bin)
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        
        # print(pred)
        # pred = P * weights
        # print(pred)

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print(probs)
        # print(probs)
        # print(log_p.size(), weights.size())

        batch_loss = -log_p * weights / tot
        # print(batch_loss)
        loss = batch_loss.sum()
        # print(loss)
        return loss


class CrossEntropyLabelAwareSmooth(CrossEntropy):
    r"""Cross entropy loss with label-aware smoothing regularizer.

    Reference:
        Zhong et al. Improving Calibration for Long-Tailed Recognition. CVPR 2021. https://arxiv.org/abs/2104.00466

    For more details of label-aware smoothing, you can see Section 3.2 in the above paper.

    Args:
        shape (str): the manner of how to get the params of label-aware smoothing.
        smooth_head (float): the largest  label smoothing factor
        smooth_tail (float): the smallest label smoothing factor
    """
    def __init__(self, para_dict=None):
        super(CrossEntropyLabelAwareSmooth, self).__init__(para_dict)

        smooth_head = self.para_dict['SMOOTH_HEAD']
        smooth_tail = self.para_dict['SMOOTH_TAIL']
        shape = self.para_dict['SHAPE']

        n_1 = max(self.num_class_list)
        n_K = min(self.num_class_list)
        if n_1 == n_K:
            n_1 = n_K + 1
        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(self.num_class_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(self.num_class_list) - n_K) / (n_1 - n_K)
        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(self.num_class_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        else:
            raise AttributeError

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        if torch.cuda.is_available():
            self.smooth = self.smooth.cuda()

    def forward(self, inputs, targets, **kwargs):
        smoothing = self.smooth[targets].to(inputs.get_device())
        confidence = 1. - smoothing
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


class Weighted_Cross_Entropy(nn.Module):

    def __init__(self, args, class_num, alpha=None, size_average=True):
        self.args = args
        super(Weighted_Cross_Entropy, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)

        inputs = inputs.float()
        P = F.log_softmax(inputs, dim=-1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(targets.get_device())
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs

        alpha = alpha.to(torch.device(self.args.gpu))
        probs = probs.to(torch.device(self.args.gpu))
        log_p = log_p.to(torch.device(self.args.gpu))

        batch_loss = - alpha * log_p
        # batch_loss = -log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class CB_loss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes, device, beta=0.99, gamma=0.5):
        super().__init__()
        samples_per_cls = [i if i != 0 else 0.1 for i in samples_per_cls]
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes

        self.weights = weights
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.no_of_classes = no_of_classes

    def forward(self, logits, labels):

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float().to(self.device)

        weights = torch.tensor(self.weights).float().to(self.device)
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, reduction="none")

        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels_one_hot * logits - self.gamma * torch.log(1 + torch.exp(-1.0 * logits))).to(self.device)

        loss = modulator * BCLoss

        weighted_loss = weights * loss
        loss = torch.sum(weighted_loss)

        loss /= torch.sum(labels_one_hot)

        return loss


class DROLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, class_weights=None, epsilons=None):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.class_weights = class_weights
        self.epsilons = epsilons

    def pairwise_euaclidean_distance(self, x, y):
        return torch.cdist(x, y)

    def pairwise_cosine_sim(self, x, y):
        x = x / x.norm(dim=1, keepdim=True)
        y = y / y.norm(dim=1, keepdim=True)
        return torch.matmul(x, y.T)

    def forward(self, batch_feats, batch_targets, centroid_feats, centroid_targets):
        device = (torch.device('cuda')
                  if centroid_feats.is_cuda
                  else torch.device('cpu'))

        classes, positive_counts = torch.unique(batch_targets, return_counts=True)
        centroid_classes = torch.unique(centroid_targets)
        train_prototypes = torch.stack([centroid_feats[torch.where(centroid_targets == c)[0]].mean(0)
                                        for c in centroid_classes])
        pairwise = -1 * self.pairwise_euaclidean_distance(train_prototypes, batch_feats)

        # epsilons
        if self.epsilons is not None:
            mask = torch.eq(centroid_classes.contiguous().view(-1, 1), batch_targets.contiguous().view(-1, 1).T).to(
                device)
            a = pairwise.clone()
            pairwise[mask] = a[mask] - self.epsilons[batch_targets].to(device)

        logits = torch.div(pairwise, self.temperature)

        # compute log_prob
        log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))
        log_prob = torch.stack([log_prob[:, torch.where(batch_targets == c)[0]].mean(1) for c in classes], dim=1)

        # compute mean of log-likelihood over positive
        mask = torch.eq(centroid_classes.contiguous().view(-1, 1), classes.contiguous().view(-1, 1).T).float().to(
            device)
        log_prob_pos = (mask * log_prob).sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * log_prob_pos
        # weight by class weight
        if self.class_weights is not None:
            weights = self.class_weights[centroid_classes]
            weighted_loss = loss * weights
            loss = weighted_loss.sum() / weights.sum()
        else:
            loss = loss.sum() / len(classes)

        return loss


###LADE###
class LADELoss(nn.Module):
    def __init__(self, num_classes=10, img_max=None, prior=None, prior_txt=None, remine_lambda=0.1):
        super().__init__()
        if img_max is not None or prior_txt is not None:
            self.img_num_per_cls = torch.Tensor(prior_txt)
            self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()
        else:
            self.prior = None

        self.balanced_prior = torch.tensor(1. / num_classes)
        self.remine_lambda = remine_lambda

        self.num_classes = num_classes
        self.cls_weight = (self.img_num_per_cls.float() / torch.sum(self.img_num_per_cls.float()))

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, y_pred, target, q_pred=None):
        """
        y_pred: N x C
        target: N
        """
        self.prior = self.prior.to(y_pred.get_device())
        self.balanced_prior = self.balanced_prior.to(y_pred.get_device())
        self.cls_weight = self.cls_weight.to(y_pred.get_device())
        per_cls_pred_spread = y_pred.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (y_pred - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)

        loss = -torch.sum(estim_loss * self.cls_weight)
        return loss


class PriorCELoss(nn.Module):
    # Also named as LADE-CE Loss
    def __init__(self, num_classes, img_max=None, prior=None, prior_txt=None):
        super().__init__()
        self.img_num_per_cls = torch.Tensor(prior_txt)
        self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x, y):
        self.prior = self.prior.to(x.get_device())
        logits = x + torch.log(self.prior + 1e-9)
        loss = self.criterion(logits, y)
        return loss


class PaCoLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=128,
                 num_classes=100, device=None):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes
        self.device = device

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(self.device)

    def forward(self, features, labels=None, sup_logits=None):
        device = self.device

        ss = features.shape[0]
        batch_size = (features.shape[0] - self.K) // 2

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits
        anchor_dot_contrast = torch.cat(((sup_logits + torch.log(self.weight + 1e-9)) / self.supt, anchor_dot_contrast),
                                        dim=1)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # add ground truth
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size, ].view(-1, ), num_classes=self.num_classes).to(
            torch.float32)
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss






import torch
import torch.nn
from abc import abstractmethod
from numpy import inf
import argparse
import collections

from .. import loss as module_loss
from ..parse_config import ConfigParser


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, optimizer, args, cls_num_list):

        self.args = args
        self.cls_num_list = cls_num_list
        config, criterion = self.init_()

        self.config, self.criterion = config, criterion
        self.metric = None

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.device_ids = device_ids
        self.model = model

        self.real_model = self.model

        self.criterion = self.criterion.to(self.device)

        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = args.epochs
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

    def init_(self):

        CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
        args = [
            CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
            CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
            CustomArgs(['--name'], type=str, target='name'),
            CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
            CustomArgs(['--step1'], type=int, target='lr_scheduler;args;step1'),
            CustomArgs(['--step2'], type=int, target='lr_scheduler;args;step2'),
            CustomArgs(['--warmup'], type=int, target='lr_scheduler;args;warmup_epoch'),
            CustomArgs(['--gamma'], type=float, target='lr_scheduler;args;gamma'),
            CustomArgs(['--save_period'], type=int, target='trainer;save_period'),
            CustomArgs(['--reduce_dimension'], type=int, target='arch;args;reduce_dimension'),
            CustomArgs(['--layer2_dimension'], type=int, target='arch;args;layer2_output_dim'),
            CustomArgs(['--layer3_dimension'], type=int, target='arch;args;layer3_output_dim'),
            CustomArgs(['--layer4_dimension'], type=int, target='arch;args;layer4_output_dim'),
            CustomArgs(['--num_experts'], type=int, target='arch;args;num_experts'),
            CustomArgs(['--distribution_aware_diversity_factor'], type=float,
                       target='loss;args;additional_diversity_factor'),
            CustomArgs(['--pos_weight'], type=float, target='arch;args;pos_weight'),
            CustomArgs(['--collaborative_loss'], type=int, target='loss;args;collaborative_loss'),
            CustomArgs(['--distill_checkpoint'], type=str, target='distill_checkpoint')
        ]

        if "ride" in self.args.method:
            config_file = "/config_imbalance_cifar100_ride.json"
        elif "ldae" in self.args.method:
            config_file = "/config_imbalance_cifar100_ldae.json"

        config = ConfigParser.from_args(args, config_file=config_file)

        # get function handles of loss and metrics
        criterion = config.init_obj('loss', module_loss, cls_num_list=self.cls_num_list, num_experts=config["arch"]["args"]["num_experts"])

        return config, criterion

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids
    
from abc import ABC, abstractmethod


class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer.
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """
    def __init__(self, model, args=None):
        # TODO: Make args mandatory during initialization
        self.model = model
        self.id = 0
        self.args = args

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def train(self, train_data, device, args=None):
        # TODO: Remove args after modifying all dependent files
        pass

    @abstractmethod
    def test(self, test_data, device, args=None):
        # TODO: Remove args after modifying all dependent files
        pass


import torch
from torchvision.utils import make_grid
import wandb





import numpy as np
import torch
from torchvision.utils import make_grid
from .base import BaseTrainer
from .utils import autocast, use_fp16


class MultiExTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, optimizer=None, data_loader=None, args=None, cls_num_list=None, training_exp=None):
        super().__init__(model, optimizer, args, cls_num_list)

        self.distill = False
        
        # add_extra_info will return info about individual experts. This is crucial for individual loss. If this is false, we can only get a final mean logits.
        self.add_extra_info = self.config._config.get('add_extra_info', False)
        if self.args.num_experts == 1:
            self.add_extra_info = False

        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.scaler = None

        # self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.training_exp = training_exp
        self.device = args.gpu
        self.model = self.model.to(self.device)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()

        if hasattr(self.criterion, "_hook_before_epoch"):
            self.criterion._hook_before_epoch(epoch)

        for batch_idx, data in enumerate(self.data_loader):
            if self.distill and len(data) == 4:
                data, target, idx, contrast_idx = data
            else:
                data, target = data

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            if self.real_model.requires_target:
                output = self.model(data, target=target)
                output, loss = output
            else:
                extra_info = {}
                output = self.model(data)

                if self.add_extra_info:
                    if isinstance(output, dict):
                        logits = output["logits"]
                        extra_info.update({
                            "logits": logits.transpose(0, 1)
                        })
                    else:
                        extra_info.update({
                            "logits": self.real_model.backbone.logits
                        })

                if isinstance(output, dict):
                    output = output["output"]

                # if self.distill:
                #     loss = self.criterion(student=output, target=target, teacher=teacher, extra_info=extra_info)
                if self.add_extra_info and "ldae" in self.args.method:
                    loss = self.criterion(output_logits=output, target=target, extra_info=extra_info, training_exp=self.training_exp, device=self.device)
                else:
                    loss = self.criterion(output_logits=output, target=target)

            loss.backward()
            self.optimizer.step()

        return

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()

        with torch.no_grad():
            if hasattr(self.model, "confidence_model") and self.model.confidence_model:
                cumulative_sample_num_experts = torch.zeros((self.model.backbone.num_experts, ), device=self.device)
                num_samples = 0
                confidence_model = True
            else:
                confidence_model = False
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                if confidence_model:
                    output, sample_num_experts = self.model(data)
                    num, count = torch.unique(sample_num_experts, return_counts=True)
                    cumulative_sample_num_experts[num - 1] += count
                    num_samples += data.size(0)
                else:
                    output = self.model(data)
                if isinstance(output, dict):
                    output = output["output"]
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, return_length=True))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if confidence_model:
                print("Samples with num_experts:", *[('%.2f'%item) for item in (cumulative_sample_num_experts * 100 / num_samples).tolist()])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def update_mode(self, model_para):
        self.model.load_state_dict(model_para)


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None):
        super().__init__(model, args)
        self.extractor_model = None
        self.class_num = None
        self.class_dist = None
        self.islt = False
        self.mixtrain_flag = True
        self.total_cls_num = None
        self.class_range = None
        self.training_exp = None


    # for long-tail dataset
    def set_ltinfo(self, class_num=None, mixtrain_flag=None, class_dist=None, class_range=None):
        if class_num is not None:
            self.class_num = class_num

        if self.args is None or "lt" in self.args.dataset:
            self.islt = True
            if mixtrain_flag is not None:
                self.mixtrain_flag = mixtrain_flag
            if class_dist is not None:
                self.class_dist = class_dist
            if class_range is not None:
                self.class_range = class_range

    def wandb_watch_model(self):
        wandb.watch(self.model)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_model(self):
        return self.model

    def set_acc_in_weight(self, cls_acc_metrics, label_smaple_num, device):

        for label in range(self.class_num):
            if label_smaple_num[label] != 0:
                cls_acc_metrics[label] = cls_acc_metrics[label] / label_smaple_num[label]

        # logging.info("cls_acc_metrics in client" + str(cls_acc_metrics))
        model_para = self.get_model_params()
        fc_weight = model_para['fc.weight']

        for i in range(self.class_num):
            fc_weight[i][0] = cls_acc_metrics[i] * 0.1

        self.set_model_params(model_para)
        self.model.to(device)

    def train(self, train_data, device, args, alpha=None, cls_num_list=None, round=0):
        model = self.model
        model.to(device)
        model.train()

        criterion, optimizer = self.train_init(device, cls_num_list, alpha)

        epoch_loss = []

        if "ride" in args.method or "ldae" in args.method:
            multiext_trainer = MultiExTrainer(model=model, optimizer=optimizer, data_loader=train_data, args=args,
                                              cls_num_list=cls_num_list, training_exp=self.training_exp)
            multiext_trainer.train()

            return

        train_data.dataset.target = train_data.dataset.target.astype(np.int64)

        cls_acc_metrics = dict.fromkeys(range(self.class_num), 0)
        label_smaple_num = dict.fromkeys(range(self.class_num), 0)

        self.train_count = 0

        for epoch in range(args.epochs):

            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                model.zero_grad()

                if isinstance(x, list):
                    x[0], x[1], labels = x[0].to(device), x[1].to(device), labels.to(device)
                    features, labels, log_probs = model(im_q=x[0], im_k=x[1], labels=labels)
                    loss = criterion(features, labels, log_probs)
                else:
                    x, labels = x.to(device), labels.to(device)
                    log_probs = model(x)

                    if "lade" in args.method:
                        perform_loss = criterion["perform"](log_probs, labels)
                        routeweight_loss = criterion["routeweight"](log_probs, labels)
                        loss = perform_loss + args.lade_weight * routeweight_loss
                    elif "ldam" in args.method:
                        loss = criterion(log_probs, labels, round, device)
                    else:
                        loss = criterion(log_probs, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

                self.train_count += 1

                if self.islt and epoch == args.epochs - 1:
                    _, predicted = torch.max(log_probs, -1)
                    correct = predicted.eq(labels)

                    for (idx, label) in enumerate(labels):
                        if correct[idx]:
                            cls_acc_metrics[int(label.item())] += 1
                        label_smaple_num[int(label.item())] += 1

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    def train_init(self, device, cls_num_list=None, alpha=None):
        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                         weight_decay=self.args.wd, amsgrad=True)

        if "focal" in self.args.method:
            # alpha = torch.Tensor([i/sum(cls_num_list) if i != 0 else 1/sum(cls_num_list) for i in cls_num_list])
            criterion = FocalLoss(gamma=0.5)
        elif "cbloss" in self.args.method:
            if "cifar10_lt" in self.args.dataset:
                beta = 0.999999
                gama = 1.0
            elif "cifar100_lt" in self.args.dataset:
                beta = 0.99
                gama = 0.8
            criterion = CB_loss(cls_num_list, self.class_num, device, beta=beta, gamma=gama)
        elif "lade" in self.args.method:
            criterion_perform = PriorCELoss(num_classes=self.class_num, prior_txt=cls_num_list).to(device)
            criterion_routeweight = LADELoss(num_classes=self.class_num, prior_txt=cls_num_list, remine_lambda=0.01).to(device)
            criterion = {"perform": criterion_perform, "routeweight": criterion_routeweight}
        elif "ldam" in self.args.method:
            criterion = LDAMLoss(cls_num_list=cls_num_list, reweight_ep=self.args.comm_round * 2/3)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        return criterion, optimizer

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
        }
        test_data.dataset.target = test_data.dataset.target.astype(np.int64)
        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)

                target = target.to(device)
                pred = model(x)

                if "lade" in self.args.method:
                    pred += torch.log(torch.ones(self.class_num)/self.class_num).to(device)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        return metrics

    def test_for_all_labels(self, test_data, device):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
            'all_preds': 0,
            'Many acc': 0,
            'Medium acc': 0,
            'Few acc': 0,
        }

        label_smaple_num = {}
        for i in range(self.class_num):
            metrics[i] = 0
            label_smaple_num[i] = 0

        test_data.dataset.target = test_data.dataset.target.astype(np.int64)
        criterion = nn.CrossEntropyLoss().to(device)

        all_preds = torch.tensor([])
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                if "lade" in self.args.method:
                    pred += torch.log(torch.ones(self.class_num)/self.class_num).to(device)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

                all_preds = torch.cat((all_preds, predicted.cpu()), dim=0)

                for (idx, label) in enumerate(target):
                    if predicted[idx].eq(target[idx]):
                        metrics[label.item()] += 1
                    label_smaple_num[label.item()] += 1

            for label in range(self.class_num):
                if label_smaple_num[label] != 0:
                    metrics[label] = metrics[label] / label_smaple_num[label]

            if self.class_range is not None:
                for i in range(self.class_num):
                    if i < self.class_range[0]:
                        metrics['Many acc'] += metrics[i]
                    elif i < self.class_range[1]:
                        metrics['Medium acc'] += metrics[i]
                    else:
                        metrics['Few acc'] += metrics[i]

                metrics['Many acc'] /= self.class_range[0]
                metrics['Medium acc'] /= self.class_range[1] - self.class_range[0]
                if metrics['Medium acc'] < 0:
                    metrics['Medium acc'] = 0
                metrics['Few acc'] /= self.class_num - self.class_range[1]
                if metrics['Few acc'] < 0:
                    metrics['Few acc'] = 0

        metrics['all_preds'] = all_preds

        return metrics



import copy
import logging
from collections import Counter
import numpy as np
import torch
import wandb
import torchvision.transforms as trans
from fedml_api.clsimb_fedavg.client import Client

class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args

        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num,
         traindata_cls_counts] = dataset

        self.all_data = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.traindata_cls_counts = traindata_cls_counts

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.class_num = class_num

        if "esti_global" in self.args.method:
            self.avg_similarity = 0
            self.total_esti_cls_num = None
            self.count = 0
            self.esti_cls_num_list = []

        if "ldae" in self.args.method or "ride" in self.args.method:
            self.experts_acc_rounds_list = []

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        total_clt_num = self.args.client_num_in_total
        w_global = self.model_trainer.get_model_params()

        if "global" in self.args.method:
            total_cls_count = Counter(self.train_global.dataset.target)

        if "esti_global" in self.args.method:
            self.estimate_local_distribution(w_global)

        oringin_lr = self.args.lr

        round_range = range(self.args.comm_round)

        for round_idx in round_range:

            logging.info("################Communication round : {}".format(round_idx))

            ###lr schedular ###
            if self.args.lr_decay == 0:
                if round_idx >= int(self.args.comm_round * 1/10):
                    self.args.lr -= (oringin_lr / self.args.comm_round)
            elif self.args.lr_decay > 0:
                if round_idx == int(self.args.comm_round * 4/5):
                    self.args.lr = oringin_lr * self.args.lr_decay
                elif "imagenet224" in self.args.dataset and round_idx == int(self.args.comm_round * 9/10) \
                        and self.args.save_load != 3:
                    self.args.lr = self.args.lr * self.args.lr_decay
                    self.args.frequency_of_the_test = 50

            if "lade" in self.args.method:
                if "blsm" in self.args.method:
                    self.args.lade_weight = 0
                elif "imagenet224" in self.args.dataset:
                    self.args.lade_weight = 0.1
                elif "cifar100" in self.args.dataset:
                    self.args.lade_weight = 0.1
                else:
                    self.args.lade_weight = 0.01

            if "train_exp" in self.args.method and round_idx >= int(self.args.comm_round * 3/5):
                self.specifical_exp(round_idx=round_idx, expert_num=self.args.num_experts)
                if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
                    self._global_test(round_idx)

                continue

            w_locals = []

            client_indexes = self._client_sampling(round_idx, total_clt_num,
                                                   self.args.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                if "real_global" in self.args.method:
                    client.set_cls_num_list(total_cls_count)
                elif "esti_global" in self.args.method:
                        client.set_cls_num_list(list(self.total_esti_cls_num))
                else:
                    client.set_cls_num_list(self.traindata_cls_counts[client_idx])

                if "ride" in self.args.method:
                    client_cls = self.traindata_cls_counts[client_idx]
                    max_cls_num = max(client_cls.values())
                    class_dist = {i: max_cls_num / client_cls[i] if i in client_cls.keys() else 0 for i in
                                  range(self.class_num)}

                    self.model_trainer.set_ltinfo(class_dist=torch.tensor(list(class_dist.values())))

                w = client.train(w_global, round=round_idx)

                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            self.model_trainer.set_model_params(w_global)

            if round_idx == self.args.comm_round - 1:
                self._global_test(round_idx)
            elif round_idx % self.args.frequency_of_the_test == 0 and round_idx != 0:
                self._global_test(round_idx)

    def _aggregate(self, w_locals, client_ratio=None, global_model=None, client_cls_list=None, round_idx=0):
        training_num = 0
        ratio_training_num = 0

        if client_ratio is None or len(client_ratio) < len(w_locals):
            client_ratio = [1] * len(w_locals)
        else:
            logging.info("client_ratio", client_ratio)

        for idx in range(len(w_locals)):
            (sample_num, local_params) = w_locals[idx]

            if "esti_global" in self.args.method and self.count < self.args.client_num_in_total:
                self.server_estimate_global_distribution(local_params, global_model, sample_num, client_cls_list, idx)

            ratio_sample_num = sample_num * client_ratio[idx]
            ratio_training_num += ratio_sample_num

            training_num += sample_num

        (sample_num, averaged_params) = copy.deepcopy(w_locals[0])
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        return averaged_params

    def server_estimate_global_distribution(self, averaged_params, global_model, sample_num, client_cls_list, idx):
        classifier_key = list(averaged_params.keys())[-1]
        if "fc" in classifier_key:
            class_dist = self._get_weight_value(averaged_params["fc.weight"] - global_model["fc.weight"],
                                                "fc.weight", method="sum")
        elif "backbone.linear" in classifier_key:
            if self.args.num_experts > 1:
                local_linear = averaged_params["backbone.linears.0.weight"]
                global_linear = global_model["backbone.linears.0.weight"]

                for i in range(1, self.args.num_experts):
                    local_linear += averaged_params["backbone.linears." + str(i) + ".weight"]
                    global_linear += global_model["backbone.linears." + str(i) + ".weight"]

                local_linear /= self.args.num_experts
                global_linear /= self.args.num_experts

                class_dist = self._get_weight_value(local_linear - global_linear, "fc.weight", method="sum")
            else:
                class_dist = self._get_weight_value(averaged_params["backbone.linear.weight"] - global_model["backbone.linear.weight"],
                                                    "backbone.linear.weight", method="sum")

        real_dist = np.array(client_cls_list[idx])

        class_dist = class_dist.numpy()
        class_dist = np.where(class_dist <= 0, 1e-5, class_dist)
        class_dist = class_dist / sum(class_dist) if sum(class_dist) != 0 else [0] * self.args.client_num_per_round
        esti_cls_num = np.around(class_dist * sample_num).astype(np.int)

        if len(self.esti_cls_num_list) < self.args.client_num_in_total:
            self.esti_cls_num_list.append(list(esti_cls_num.astype(int)))

        similarity = cosine_similarity(esti_cls_num, real_dist)

        if self.total_esti_cls_num is None:
            self.total_esti_cls_num = copy.deepcopy(esti_cls_num)
            self.avg_similarity = similarity
            self.count = 1

        elif self.count < self.args.client_num_in_total:
            self.total_esti_cls_num += esti_cls_num
            # logging.info(str(esti_cls_num) + str(self.count) + "--Estimate\n" + str(real_dist) + "--Real; similarity:" + str(similarity))
            self.count += 1
            self.avg_similarity += similarity
            if self.count == self.args.client_num_in_total:
                self.total_esti_cls_num = [i if i >= 5 else 5 for i in self.total_esti_cls_num.astype(np.int)]

                # logging.info("Total_esti_cls_num: " + str(self.total_esti_cls_num))
                if "imagenet" in self.args.dataset:
                    real_total_cls_num = []
                    for label in range(self.class_num):
                        real_total_cls_num.append(0)
                        for clt in self.traindata_cls_counts.values():
                            if label in clt.keys():
                                real_total_cls_num[label] += clt[label]
                    real_total_cls_num = np.array(real_total_cls_num)
                else:
                    real_total_cls_num = np.array(list(Counter(self.train_global.dataset.target).values()))

                self.avg_similarity /= self.count
                logging.info("Avg similarity:" + str(self.avg_similarity))

                total_similarity = cosine_similarity(self.total_esti_cls_num, real_total_cls_num)
                logging.info("Total num similarity:" + str(total_similarity))

    def estimate_local_distribution(self, init_model):

        real_ep = self.args.epochs
        real_lr = self.args.lr
        real_bs = self.args.batch_size
        real_method = self.args.method

        self.args.epochs = 1

        self.args.method = "esti_global"

        client_cls_list = []
        w_locals = []

        client = self.client_list[0]

        esti_round = 1
        w_global = init_model

        client_indexes = self._client_sampling(0, self.args.client_num_in_total, self.args.client_num_in_total)

        for round in range(esti_round):
            for client_idx in client_indexes:
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                client.set_cls_num_list(self.traindata_cls_counts[client_idx])

                client_cls_dic = self.traindata_cls_counts[client_idx]
                # for calculate similarity
                client_cls_list.append(
                    [client_cls_dic[idx] if idx in client_cls_dic.keys() else 0 for idx in range(self.class_num)])

                w = client.train(w_global)

                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
            w_global = self._aggregate(w_locals, global_model=w_global, client_cls_list=client_cls_list)
            w_old_global = w_global

        for client_idx in client_indexes:
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])

            client.set_cls_num_list(self.traindata_cls_counts[client_idx])

            client_cls_dic = self.traindata_cls_counts[client_idx]
            # for calculate similarity
            client_cls_list.append(
                [client_cls_dic[idx] if idx in client_cls_dic.keys() else 0 for idx in range(self.class_num)])

            w = client.train(w_old_global)

            w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
        self._aggregate(w_locals, global_model=w_old_global, client_cls_list=client_cls_list)

        self.args.epochs = real_ep
        self.args.lr = real_lr

        self.args.batch_size = real_bs
        self.args.method = real_method
        # self.args.client_num_per_round = real_client_num_per_round

    def specifical_exp(self, round_idx, expert_num):

        client_ratio = [self.args.beta] * expert_num

        clt_ratio_list = []
        if "real_global" in self.args.method:
            total_cls_count = np.array(list(Counter(self.train_global.dataset.target).values()))

            for (i, clt_cls_num) in self.traindata_cls_counts.items():
                clt_cls_num = np.array([clt_cls_num[i] if i in clt_cls_num.keys() else 0 for i in range(self.class_num)])
                clt_cls_num = np.array(clt_cls_num/sum(clt_cls_num))
                clt_ratio_list.append((sum(total_cls_count * clt_cls_num), i))

        elif "esti_global" in self.args.method:
            for (i, clt_cls_num) in enumerate(self.esti_cls_num_list):
                clt_cls_num = np.array(clt_cls_num/sum(clt_cls_num))
                clt_ratio_list.append((sum(self.total_esti_cls_num * clt_cls_num), i))

        clt_ratio_list.sort(reverse=True, key=lambda x: x[0])

        clts_sorted_by_cls = [i[1] for i in clt_ratio_list]
        # logging.info("###########clts_sorted_by_cls########\n" + str(clts_sorted_by_cls))
        expert_clt_idxs = []
        for i in range(expert_num):
            expert_clt_idxs.append(clts_sorted_by_cls[int(self.args.client_num_in_total * i/expert_num):int(self.args.client_num_in_total* (i+1)/expert_num)])

        for i in range(expert_num):
            if self.args.client_num_per_round * self.args.beta >= len(expert_clt_idxs[i]):
                clt_indexes = expert_clt_idxs[i]
                rand_expert_num = self.args.client_num_per_round * self.args.beta - len(clt_indexes)
                random_clt = np.random.choice(range(self.args.client_num_in_total), rand_expert_num, replace=False)
                # clt_indexes = expert_clt_idxs[2]
            else:
                np.random.seed(round_idx + 1)
                clt_indexes = np.random.choice(expert_clt_idxs[i], int(self.args.client_num_per_round * client_ratio[i]), replace=False)
                # clt_indexes = np.random.choice(expert_clt_idxs[2], int(self.args.client_num_per_round * client_ratio[i]), replace=False) ##rare client train
                random_clt = np.random.choice(range(self.args.client_num_in_total), int(self.args.client_num_per_round * (1-client_ratio[i])), replace=False)

            clt_indexes = np.concatenate((clt_indexes, random_clt))
            logging.info("expert :{0} client_indexes = :{1}".format(i, str(clt_indexes)))

            self.experts_train(round_idx=round_idx, train_experts=str(i), clt_indexes=clt_indexes)

    def experts_train(self, round_idx, train_experts=None, clt_indexes=None):
        w_global = self.model_trainer.get_model_params()
        w_locals = []

        for idx in range(len(clt_indexes)):
            # update dataset
            client = self.client_list[idx]
            client_idx = clt_indexes[idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])

            # client.set_cls_num_list(self.traindata_cls_counts[client_idx]) #local
            if "real_global" in self.args.method:
                client.set_cls_num_list(Counter(self.train_global.dataset.target))
            elif "esti_global" in self.args.method:
                client.set_cls_num_list(list(self.total_esti_cls_num))
            else:
                client.set_cls_num_list(self.traindata_cls_counts[client_idx])

            self.freeze_layer(train_experts)

            w = client.train(w_global, round=round_idx)

            self.freeze_layer()

            w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

        w_global = self._aggregate(w_locals, global_model=w_global)
        self.model_trainer.set_model_params(w_global)

    def freeze_layer(self, train_experts=None):
        self.model_trainer.training_exp = train_experts

        model = self.model_trainer.model
        if train_experts is not None:
            for name, para in model.named_parameters():
                if "s." in name and ("s." + train_experts) not in name:
                    para.requires_grad = False
                else:
                    para.requires_grad = True
        else:
            for name, para in model.named_parameters():
                para.requires_grad = True


    def _get_weight_value(self, param, round_idx=0, method=None):
        if param.shape[0] == self.class_num:
            dim = 1
        else:
            dim = 0

        if "abs_sum" in method:
            if len(param.shape) == 1:
                para_norm = abs(param)
            else:
                para_norm = abs(param)
                para_norm = para_norm.sum(0)
        elif "norm" in method:
            norm = torch.norm(param, 2, 1)
            if len(norm.shape) == 1:
                para_norm = norm
            else:
                para_norm = norm.sum(1)
        elif "min" in method:
            param = torch.min(param, dim=-1)
            para_norm = abs(param.values)
        elif "sum" in method:
            para_norm = torch.sum(param, dim=dim)
        else:
            logging.warning("No such Weight Value")
            return

        return para_norm

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx + 1)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        return client_indexes

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):

            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])

            # train data
            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            if 0 in train_local_metrics.keys():
                for i in range(self.class_num):
                    if i not in train_metrics.keys():
                        train_metrics[i] = []
                        test_metrics[i] = []
                    train_metrics[i].append(copy.deepcopy(train_local_metrics[i]))
                    test_metrics[i].append(copy.deepcopy(test_local_metrics[i]))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

        if 0 in train_metrics.keys():
            lable_acc = {}
            for i in range(self.class_num):
                lable_acc[i] = sum(train_metrics[i]) / self.args.client_num_in_total
            # logging.info("train_acc_per_label:" + str(lable_acc))

            for i in range(self.class_num):
                lable_acc[i] = sum(test_metrics[i]) / self.args.client_num_in_total
            logging.info("test_acc_per_label:" + str(lable_acc))

    def _global_test(self, round_idx):

        logging.info("################global_test###################")


        train_metrics = {
            'test_correct': 1,
            'test_total': 1,
            'test_loss': 1
        }

        test_metrics = {
            'num_samples': 1,
            'num_correct': 1,
            'losses': 1
        }

        # train data
        if "imagenet" in self.args.dataset:
            train_local_metrics = train_metrics
        else:
            train_local_metrics = self.model_trainer.test(self.train_global, self.device, self.args)

        train_acc = copy.deepcopy(train_local_metrics['test_correct'])
        train_num = copy.deepcopy(train_local_metrics['test_total'])
        train_loss = copy.deepcopy(train_local_metrics['test_loss'])

        train_acc = train_acc / train_num
        train_loss = train_loss / train_num

        test_local_metrics = self.model_trainer.test(self.test_global, self.device, self.args)

        test_acc = copy.deepcopy(test_local_metrics['test_correct'])
        test_num = copy.deepcopy(test_local_metrics['test_total'])
        test_loss = copy.deepcopy(test_local_metrics['test_loss'])

        test_acc = test_acc / test_num
        test_loss = test_loss / test_num

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

        train_lable_matric = self.model_trainer.test_for_all_labels(self.train_global, self.device)
        test_label_matric = self.model_trainer.test_for_all_labels(self.test_global, self.device)

        train_label_acc = {}
        test_label_acc = {}
        for i in range(self.class_num):
            train_label_acc[i] = train_lable_matric[i]
            test_label_acc[i] = test_label_matric[i]
        #     wandb.log({"F1score class " + str(i): f1[i], "round": round_idx})
        wandb.log({"Many-shot acc ": test_label_matric["Many acc"], "round": round_idx})
        wandb.log({"Medium-shot acc ": test_label_matric["Medium acc"], "round": round_idx})
        wandb.log({"Few-shot acc ": test_label_matric["Few acc"], "round": round_idx})

        logging.info("train_acc_per_label:" + str(train_label_acc))
        logging.info("Many-shot acc:" + str(test_label_matric["Many acc"]))
        logging.info("Medium-shot acc:" + str(test_label_matric["Medium acc"]))
        logging.info("Few-shot acc:" + str(test_label_matric["Few acc"]))

    #### delete one of samples if there is one redundacy sample, due to the drop_last=True
    def fix_redundacy_sample(self, batch_size, train_data_local_dict):
        for idx in train_data_local_dict:
            data = train_data_local_dict[idx].dataset.data
            target = train_data_local_dict[idx].dataset.target
            redundacy = data.shape[0] % batch_size
            if redundacy == 1:
                logging.info("delete one sample to avoid bug in batch_norm for client " + str(idx))
                train_data_local_dict[idx] = torch.utils.data.DataLoader(dataset=train_data_local_dict[idx].dataset,
                                                                             batch_size=batch_size, shuffle=True,
                                                                             drop_last=True)


def cosine_similarity(x, y):
    similarity = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    return similarity




python -u ./main_fedavg.py \
--lr 0.6 --lr_decay 0.05 \
--method ldae_train_exp_esti_global --frequency_of_the_test 50 --beta 0.8
