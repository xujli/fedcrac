import torch
import wandb
from methods.base import Base_Client, Base_Server

import json
import logging
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.nn.modules.loss import _WeightedLoss
from torch.multiprocessing import current_process
from torch.autograd.function import Function
from sklearn.metrics import confusion_matrix
import math
from copy import deepcopy
import cv2

import torch
from torch.autograd import Variable
from collections import OrderedDict
from models.MarginLinear import SoftmaxMargin, SoftmaxMarginL
def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c

def sinkhorn_loss(x, y, epsilon, n, niter, device):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    # The Sinkhorn algorithm takes as input three variables :
    C = Variable(cost_matrix(x, y)).to(device)  # Wasserstein cost function

    # both marginals are fixed with equal weights
    # mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    # nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    mu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False).to(device)
    nu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False).to(device)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).data.cpu().numpy():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return cost



def sinkhorn_normalized(x, y, epsilon, n, niter, device):

    Wxy = sinkhorn_loss(x, y, epsilon, n, niter, device)
    Wxx = sinkhorn_loss(x, x, epsilon, n, niter, device)
    Wyy = sinkhorn_loss(y, y, epsilon, n, niter, device)
    return 2 * Wxy - Wxx - Wyy


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)


def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

    cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost

class LGMLoss_v0(torch.nn.Module):
    """
    LGMLoss whose covariance is fixed as Identity matrix
    """

    def __init__(self, num_classes, feat_dim, alpha=1.0):
        super(LGMLoss_v0, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))
        torch.nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))

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

class LGMLoss(torch.nn.Module):
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

        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))
        self.log_covs = torch.nn.Parameter(torch.zeros(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        log_covs = torch.unsqueeze(self.log_covs, dim=0)


        covs = torch.exp(log_covs) # 1*c*d
        tcovs = covs.repeat(batch_size, 1, 1) # n*c*d
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1) #eq.(18)


        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.alpha)
        y_onehot = y_onehot + 1.0
        margin_dist = torch.mul(dist, y_onehot)

        slog_covs = torch.sum(log_covs, dim=-1) #1*c
        tslog_covs = slog_covs.repeat(batch_size, 1)
        margin_logits = -0.5*(tslog_covs + margin_dist) #eq.(17)
        logits = -0.5 * (tslog_covs + dist)

        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        cdist = cdiff.pow(2).sum(1).sum(0) / 2.0

        slog_covs = torch.squeeze(slog_covs)
        reg = 0.5*torch.sum(torch.index_select(slog_covs, dim=0, index=label.long()))
        likelihood = (1.0/batch_size) * (cdist + reg)

        return logits, margin_logits, likelihood

class CenterLoss(torch.nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None

in_features_dict = {
    'Lenet5': 84,
    'OneDCNN': 256,
    'SimpleCNN': 84,
    'modVGG': 512,
    'resnet10': 512,
    'resnet18': 512,
    'resnet56': 2048
}


def loss_fn_kd(outputs, labels, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = 0.5
    T = 0.5
    KD_loss = F.kl_div(F.log_softmax(outputs / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True, margin=args.mu).to(self.device)
        self.clf_global = torch.clone(self.model.clf.weight.data.detach()).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, #momentum=0.9 if args.momentum else 0,# nesterov=True,
                                         weight_decay=self.args.wd)

        # self.optimzer4center = torch.optim.SGD(self.likelihood.parameters(), lr=0.01, momentum=0.9)
        self.global_model = deepcopy(self.model).to(self.device)
        self.eposilon = 0.01
        self.alpha = 0.5
        self.centers = None
        self.var = None

    def load_client_state_dict(self, received_info):
        server_state_dict = received_info[0]
        self.model.load_state_dict(server_state_dict)
        self.model.clf.margin = torch.clone(received_info[1]).to(self.device)

    def get_cdist(self):
        dist = self.client_cnts / self.client_cnts.sum()  # 个数的比例
        cdist = dist / dist.max()  #
        cdist = cdist * (1.0 - self.alpha) + self.alpha
        cdist = cdist.reshape((1, -1))

        return cdist.to(self.device)

    def run(self, received_info):
        client_results = []
        for client_idx in self.client_map[self.round]:
            self.load_client_state_dict(received_info)
            # self.model.clf.margin = (received_info['distances'][client_idx] * self.args.mu).to(self.device)
            # self.centers = received_info[1]
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.batch_size
            weights = self.train()
            acc = self.test()
            dist = self.init_client_infos()
            client_results.append(
                {'weights': weights, 'num_samples': num_samples, 'acc': acc,
                 'client_index': self.client_index, 'dist': dist,
                 'mean': self.centers, 'cov': self.var}
            )
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        # self.tsne_vis()
        return client_results

    def train(self):
        # train the local model
        cdist = self.get_dist().to(self.device)
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []

        hs = None
        labelss = None
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                # self.optimzer4center.zero_grad()
                h, logits = self.model(images, labels)

                # unique_labels = torch.unique(labels)
                if self.centers is None:
                    self.centers = np.zeros((self.num_classes, h.shape[1]))
                    self.var = np.zeros((self.num_classes, h.shape[1]))
                # for label in unique_labels:
                #     selected_hs = h[labels == label, :]
                #     centers[label] += torch.sum(selected_hs / (torch.norm(selected_hs, dim=1, keepdim=True)),
                #                                 dim=0).detach()
                #     counts[label] += len(selected_hs)
                # centerloss = self.likelihood(h, labels)
                # g_h, g_output = self.global_model(images, labels)
                # cost = sinkhorn_normalized(probs, g_output, self.eposilon, h.shape[0], self.n_iter, self.device)
                loss = self.criterion(logits, labels)
                       # 0.1 * F.kl_div(F.log_softmax(h, dim=1),
                       #                                         F.softmax(g_h, dim=1),
                       #                                         reduction='batchmean')  # + 1 / phi
                loss.backward()

                self.optimizer.step()
                # self.optimzer4center.step()
                batch_loss.append(loss.item())
                hs = h.detach() if hs is None else torch.cat([hs, h.detach().clone()], dim=0)
                labelss = labels if labelss is None else torch.cat([labelss, labels.clone()], dim=0)
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                    epoch, sum(
                            epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        # self.get_discriminability(hs, labelss)

        # indices = torch.not_equal(counts, 0)
        # centers[indices] = centers[indices] / counts[indices].unsqueeze(1)
        # distance = torch.cat(
        #     [torch.mean(torch.sum(torch.pow(centers[i] - centers[indices], 2), dim=0).cpu() / (len(indices) - 1), dim=0, keepdim=True) for i in
        #      range(len(counts))], dim=0).reshape((self.num_classes))
        # distance[torch.equal(counts, torch.zeros(centers.shape[1]).to(self.device))].fill_(0)
        # distance[indices] = distance[indices] / torch.mean(distance[indices], dim=0)
        # print(distance.shape)

        # estimate
        features = None
        labelss = None
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                h, log_probs = self.model(images)
                if features is None:
                    features = torch.clone(h.detach().cpu())
                    labelss = labels.cpu()
                else:
                    features = torch.cat([features, torch.clone(h.detach().cpu())])
                    labelss = torch.cat([labelss, labels.cpu()])

        for index in range(self.num_classes):
            if torch.sum((labelss == index)) > 1:
                selected_features = features[labelss == index]
                self.centers[index, :] = np.mean(selected_features.numpy(), axis=0)
                self.var[index, :] = np.var(selected_features.numpy(), axis=0)
                # self.cov[index, :, :].add_(torch.eye(features.shape[1]) * 0.001)
        weights = self.model.cpu().state_dict()
        return weights  # , distance

    def test(self):
        self.model.to(self.device)
        self.model.eval()
        hs = None
        labels = None
        wandb_dict = {}
        test_correct = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)
                # labels.extend(target)
                h, pred = self.model(x)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number) * 100
            wandb_dict[self.args.method + "_clinet:{}_acc".format(self.client_index)] = acc
            logging.info(
                "************* Round {} Client {} Acc = {:.2f} **************".format(self.round, self.client_index,
                                                                                      acc))
        return acc



class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True, margin=args.mu)
        wandb.watch(self.model)
        self.likelihood = CenterLoss(self.num_classes, in_features_dict[self.args.net]).to(self.device)
        self.mu = args.mu

    def start(self):
        with open('{}/config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [[self.model.cpu().state_dict(), self.args.mu * torch.ones((self.num_classes, self.num_classes))] for \
                _ in range(self.args.thread_number)]

    def run(self, received_info):
        server_outputs = self.operations(received_info)
        acc = self.test()
        self.log_info(received_info, acc)
        if acc > self.acc:
            torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc
        # for x in received_info:
        #     self.distances[x['client_index']] = x['centers']

        return server_outputs

    def get_mean_var(self, client_info):
        # calculate global mean and covariance
        self.client_dist = torch.stack([c['dist'] for c in client_info]).numpy()  # num_client, num_classes
        centers = np.stack([c['mean'] for c in client_info]) # num_client, num_classes, num_features
        covs = np.stack([c['cov'] for c in client_info])  # num_client, num_classes, num_features
        self.mean = np.zeros(centers[0].shape)
        self.var = np.zeros(covs[0].shape) # num_classes, num_features
        for c in range(self.var.shape[0]):
            if np.sum(self.client_dist[:, c]) > 0:
                self.mean[c] = np.sum(centers[:, c] * np.expand_dims(self.client_dist[:, c], 1), axis=0) \
                               / np.sum(self.client_dist[:, c])
                self.var[c] = np.sum(covs[:, c] * np.expand_dims(self.client_dist[:, c], 1), axis=0) \
                               / np.sum(self.client_dist[:, c])

            # self.covs[c] = (
            #     torch.sum(covs[:, c, :, :] * (self.client_dist[:, c].unsqueeze(-1).unsqueeze(-1) - 1), dim=0) + \
            #     torch.sum(torch.stack([torch.matmul(centers[i, c, :].unsqueeze(1), centers[i, c, :].unsqueeze(0)) * \
            #                            (self.client_dist[i, c]) for i in range(covs.shape[0])], dim=0), dim=0) - \
            #     torch.matmul(self.mean[c].unsqueeze(1), self.mean[c].unsqueeze(0)) * torch.sum(self.client_dist[:, c])) \
            #         / (torch.sum(self.client_dist[:, c]) - 1)
            # self.covs[c].add_(torch.eye(self.covs.shape[-1]) * 10)
            #if np.sum(self.client_dist[:, c]) > 1:
            #    self.var[c] = (
            #        np.sum(np.stack([covs[i, c] * (self.client_dist[i, c] - 1) for i in range(covs.shape[0])], axis=0), axis=0) + \
            #        np.sum(np.stack([np.expand_dims(centers[i, c], 1) * np.expand_dims(centers[i, c], 0) * (self.client_dist[i, c]) for \
            #                               i in range(covs.shape[0])], axis=0), axis=0) - \
            #        np.expand_dims(self.mean[c], 1) * np.expand_dims(self.mean[c], 0) * np.sum(self.client_dist[:, c])) \
            #            / (np.sum(self.client_dist[:, c]) - 1)
            # self.covs[c].add_(torch.eye(self.covs.shape[-1]) * 10)
        self.mean = torch.from_numpy(self.mean).float()
        self.var = torch.from_numpy(self.var).float()
        self.client_dist = torch.from_numpy(self.client_dist).long()

        sim = torch.zeros((self.num_classes, self.num_classes))
        # inter class similarity
        for i in range(self.num_classes):
            sim[i, :] = F.cosine_similarity(self.mean[i], \
                self.mean, dim=1)

        return sim


    def operations(self, client_info):

        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]

        ssd = self.model.state_dict()
        for key in ssd:
            # if 'clf' in key:
            #     labels_sum = torch.zeros(clients_label_dist[0].shape)
            #     ssd[key] = torch.zeros(ssd[key].shape)
            #     for label_dist, sd in zip(clients_label_dist, client_sd):
            #         ssd[key] += label_dist.unsqueeze(1) * sd[key]
            #         labels_sum += label_dist
            #
            #     ssd[key] = ssd[key] / labels_sum.unsqueeze(1)
            # else:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])
        sim = self.get_mean_var(client_info)
        self.sim = (torch.sum(sim) - sim.shape[0]) / sim.shape[0] / (sim.shape[0] - 1) * self.args.mu
        print(self.sim)
        linear = self.retrain()
        sd = {}
        sd['weight'] = linear.state_dict()['weight']
        #sd.popitem('weight')
        # ssd['clf.weight'] = torch.clone(linear.weight.data.cpu())
        # ssd['clf.bias'] = torch.clone(linear.bias.data.cpu())
        self.model.load_state_dict(ssd)
        self.model.clf.load_state_dict(sd)
        # self.model.clf.margin = deepcopy(self.mu)
        # if self.args.save_client:
        #     for client in client_info:
        #         torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        self.round += 1
        return [[self.model.cpu().state_dict(), deepcopy(torch.tensor(self.mu))] for _ in range(self.args.thread_number)]


    def retrain(self):
        linear = SoftmaxMargin(in_features_dict[self.args.net], self.num_classes, margin=self.mu).to(self.device)
        linear.load_state_dict(self.model.clf.state_dict())
        optimizer = torch.optim.SGD(linear.parameters(), lr=self.args.lr,
                                    momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        linear_g = SoftmaxMargin(in_features_dict[self.args.net], self.num_classes).to(self.device)
        linear_g.load_state_dict(self.model.clf.state_dict())
        linear_g.margin = torch.tensor(self.mu)
        total_dist = torch.sum(self.client_dist, dim=0)
        virtual_representations = torch.zeros((int(torch.sum(total_dist).numpy()), self.mean.shape[-1]))
        virtual_labels = torch.zeros(int(torch.sum(total_dist).numpy())).long()
        cumsum = np.concatenate([[0], np.cumsum(total_dist.numpy())])

        criterion = torch.nn.CrossEntropyLoss()
        for i in range(len(cumsum) - 1):
            dist = np.random.multivariate_normal(self.mean[i].numpy(), np.diag(self.var[i].numpy()), size=int(cumsum[i+1] - cumsum[i]))
            virtual_representations[int(cumsum[i]): int(cumsum[i+1])] = torch.tensor(dist)
            virtual_labels[int(cumsum[i]): int(cumsum[i+1])] = i

        for epoch in range(self.args.crt_epoch):
            # logging.info(images.shape)
            images, labels = virtual_representations.to(self.device), virtual_labels.to(self.device)
            optimizer.zero_grad()
            logits = linear(images, labels)
            logits_g = linear_g(images, labels)
            loss = loss_fn_kd(logits, labels, logits_g)
            loss.backward()
            optimizer.step()

        return linear
