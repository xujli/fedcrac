import torch
import wandb
import torch.nn.functional as F
from methods.base import Base_Client, Base_Server

import math
import logging
import numpy as np

from torch.multiprocessing import current_process
from torch.utils.data import TensorDataset, DataLoader
in_features_dict = {
    'modVGG': 512,
    'SimpleCNN': 84,
    'resnet10': 512,
    'resnet18': 512,
    'resnet56': 2048
}

def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)

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

        covs = torch.exp(log_covs)  # 1*c*d
        tcovs = covs.repeat(batch_size, 1, 1)  # n*c*d
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1)  # eq.(18)

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = torch.Variable(y_onehot).cuda()
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

class LGMLoss_v0(torch.nn.Module):
    """
    LGMLoss whose covariance is fixed as Identity matrix
    """

    def __init__(self, num_classes, feat_dim, alpha):
        super(LGMLoss_v0, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

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

def torch_cov(input_vec:torch.tensor):
    x = input_vec - torch.mean(input_vec, dim=0)
    cov_matrix = torch.matmul(x.T, x) / (x.shape[0]-1)
    return cov_matrix

def diag(vector):
    mat = np.zeros((vector.shape[0], vector.shape[0]))
    for i in range(vector.shape[0]):
        mat[i, i] = vector[i]
    return mat

class AutoE(torch.nn.Module):
    def __init__(self, in_feat, out_feat):
        super(AutoE, self).__init__()
        self.linear1 = torch.nn.Linear(in_feat, out_feat)
        self.linear2 = torch.nn.Linear(out_feat, in_feat)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,
                                         weight_decay=self.args.wd)
        self.eps = 1e-3
        self.linear = torch.nn.Linear(self.num_classes, in_features_dict[self.args.net])

        self.centers = None
        self.var = None

    def get_margin(self):
        self.client_cnts = self.init_client_infos()
        self.T = self.args.mu
        self.model.clf.margin = self.T / torch.pow(self.client_cnts, 1 / 4).to(self.device)

    def load_client_state_dict(self, server_state_dict):
        # If you want to customize how to state dict is loaded you can do so here
        if self.round == 0:
            super(Client, self).load_client_state_dict(server_state_dict)
        else:
            self.model.load_state_dict(server_state_dict[0])
            self.linear.load_state_dict(server_state_dict[1])

    def run(self, received_info):
        client_results = []
        for client_idx in self.client_map[self.round]:
            self.load_client_state_dict(received_info)
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.batch_size
            weights = self.train()
            acc = self.test()
            client_results.append(
                {'weights': weights, 'num_samples': num_samples, 'acc': acc, 'client_index': self.client_index,
                 'dist': self.init_client_infos(), 'mean': self.centers, 'cov': self.var})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        # self.tsne_vis()
        return client_results

    def reverse_sampling(self):

        random_sample = torch.randn(10000, self.num_classes).to(self.device)
        random_sample = random_sample / torch.sum(random_sample, dim=1)
        sample_labels = torch.argmax(random_sample, dim=1)
        rev_input = F.linear(random_sample - rev_bias, rev_linear)
        means = torch.zeros(self.num_classes, rev_input.shape[1]).to(self.device)
        cov = torch.zeros(self.num_classes, rev_input.shape[1], rev_input.shape[1]).to(self.device)
        for i in range(self.num_classes):
            means[i] = torch.mean(rev_input[sample_labels == i], dim=0)
            cov[i] = torch.cov(rev_input[sample_labels == i].T)
        return means, cov


    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        self.linear.to(self.device)
        opt = torch.optim.SGD(self.linear.parameters(), lr=0.01)
        # self.get_margin()
        epoch_loss = []
        # if self.round != 0:
        #     self.centers, self.cov = self.reverse_sampling()

        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                h, log_probs = self.model(images)
                # log_probs = log_probs / torch.norm(self.model.clf.weight.data.detach(), 1)
                if self.round != 0:
                    opt.zero_grad() # only erase gradient
                    # cdiff = h - torch.index_select(self.centers, dim=0, index=labels)
                    # likelihood = (1.0 / images.shape[0]) * cdiff.pow(2).mean(1).sum(0) / 2.0
                    mse = F.l1_loss(h, self.linear(log_probs))
                    loss = self.criterion(log_probs, labels) + 1 * mse
                else:
                    loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                    epoch, sum(
                            epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))

        if self.round != 0:
            print(mse.item())
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
                if self.centers is None:
                    self.centers = np.zeros((self.num_classes, features.shape[1]))
                    self.var = np.zeros((self.num_classes, features.shape[1], features.shape[1]))
                self.centers[index, :] = np.mean(selected_features.numpy(), axis=0)
                self.var[index, :, :] = np.cov(selected_features.numpy().T)
        weights = self.model.cpu().state_dict()
        return weights

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
        self.model = self.model_type(self.num_classes, KD=True)
        wandb.watch(self.model)
        self.mean = None
        self.var = None
        self.AE = AutoE(in_features_dict[self.args.net], self.num_classes)
        self.AEoptimizer = torch.optim.Adam([{'params': self.AE.linear1.parameters(), 'lr': 0}, {'params': self.AE.linear2.parameters()}],
                                    lr=self.args.lr, weight_decay=self.args.wd)

    # def operations(self, client_info):
    #
    #     client_info.sort(key=lambda tup: tup['client_index'])
    #     clients_label_dist = [c['dist'] for c in client_info]
    #     client_sd = [c['weights'] for c in client_info]
    #     cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]
    #
    #     ssd = self.model.state_dict()
    #     for key in ssd:
    #         if 'clf' in key:
    #             labels_sum = torch.zeros(clients_label_dist[0].shape)
    #             ssd[key] = torch.zeros(ssd[key].shape)
    #             for label_dist, sd in zip(clients_label_dist, client_sd):
    #                 ssd[key] += label_dist.unsqueeze(1) * sd[key]
    #                 labels_sum += label_dist
    #
    #             ssd[key] = ssd[key] / labels_sum.unsqueeze(1)
    #         else:
    #             ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])
    #
    #     self.model.load_state_dict(ssd)
    #     if self.args.save_client:
    #         for client in client_info:
    #             torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
    #     self.round += 1
    #     return [self.model.cpu().state_dict() for _ in range(self.args.thread_number)]
    
    def operations(self, client_info):

        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]

        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])

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

            # self.covs[c] = (
            #     torch.sum(covs[:, c, :, :] * (self.client_dist[:, c].unsqueeze(-1).unsqueeze(-1) - 1), dim=0) + \
            #     torch.sum(torch.stack([torch.matmul(centers[i, c, :].unsqueeze(1), centers[i, c, :].unsqueeze(0)) * \
            #                            (self.client_dist[i, c]) for i in range(covs.shape[0])], dim=0), dim=0) - \
            #     torch.matmul(self.mean[c].unsqueeze(1), self.mean[c].unsqueeze(0)) * torch.sum(self.client_dist[:, c])) \
            #         / (torch.sum(self.client_dist[:, c]) - 1)
            # self.covs[c].add_(torch.eye(self.covs.shape[-1]) * 10)
            if np.sum(self.client_dist[:, c]) > 1:
                self.var[c] = (
                    np.sum(np.stack([covs[i, c] * (self.client_dist[i, c] - 1) for i in range(covs.shape[0])], axis=0), axis=0) + \
                    np.sum(np.stack([np.expand_dims(centers[i, c], 1) * np.expand_dims(centers[i, c], 0) * (self.client_dist[i, c]) for \
                                           i in range(covs.shape[0])], axis=0), axis=0) - \
                    np.expand_dims(self.mean[c], 1) * np.expand_dims(self.mean[c], 0) * np.sum(self.client_dist[:, c])) \
                        / (np.sum(self.client_dist[:, c]) - 1)
            # self.covs[c].add_(torch.eye(self.covs.shape[-1]) * 10)

        self.mean = torch.from_numpy(self.mean).float()
        self.var = torch.from_numpy(self.var).float()
        self.client_dist = torch.from_numpy(self.client_dist).long()
        linear = self.retrain()
        feature_net_params = linear.state_dict()
        ssd['clf.weight'] = torch.clone(feature_net_params['weight'].cpu())
        ssd['clf.bias'] = torch.clone(feature_net_params['bias'].cpu())
        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        self.round += 1
        return [[self.model.cpu().state_dict(), self.AE.linear2.cpu().state_dict()] for _ in range(self.args.thread_number)]

    def reverse_sampling(self):
        AE = torch.nn.Sequential(
            torch.nn.Linear(self.model.clf.weight.shape[1], self.model.clf.weight.shape[0]),
            torch.nn.Linear(self.model.clf.weight.shape[0], self.model.clf.weight.shape[1])
        )
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(AE.parameters(), lr=self.args.lr, momentum=0.9,
                                         weight_decay=self.args.wd, nesterov=True)
        ssd = AE.cpu().state_dict()
        ssd['0.weight'] = self.model.clf.weight.detach().clone()
        ssd['0.bias'] = self.model.clf.bias.detach().clone()
        AE.load_state_dict(ssd)
        for name, params in AE.named_parameters():
            if '0' in name:
                params.requires_grad = False

        AE.to(self.device)
        random_sample = torch.distributions.Uniform()(10000, self.model.clf.weight.shape[1]).to(self.device) # 覆盖的范围不够？

        for epoch in range(self.args.crt_epoch):
            # logging.info(images.shape)
            optimizer.zero_grad()
            probs = AE(random_sample)
            loss = loss_func(probs, random_sample)
            loss.backward()
            optimizer.step()

        ssd = AE.cpu().state_dict()
        linear = torch.nn.Linear(self.model.clf.weight.shape[0], self.model.clf.weight.shape[1])
        linear.weight.data = ssd['1.weight'].cpu().detach().clone()
        linear.weight.bias = ssd['1.bias'].cpu().detach().clone()
        linear.to(self.device)
        random_sample = torch.randn(10000, self.num_classes).to(self.device) # 覆盖的范围不够？
        # random_sample = random_sample / torch.sum(random_sample, dim=1)
        sample_labels = torch.argmax(random_sample, dim=1)
        with torch.no_grad():
            rev_input = linear(random_sample)
        # means = torch.zeros(self.num_classes, rev_input.shape[1])
        # cov = torch.zeros(self.num_classes, rev_input.shape[1], rev_input.shape[1])
        # for i in range(self.num_classes):
        #     means[i] = torch.mean(rev_input[sample_labels == i], dim=0)
        #     cov[i] = torch.cov(rev_input[sample_labels == i])
        return rev_input, sample_labels, linear

    def retrain(self):
        linear = torch.nn.Linear(in_features_dict[self.args.net], self.num_classes).to(self.device)
        optimizer = torch.optim.SGD(linear.parameters(), lr=self.args.lr, momentum=0.9,
                                         weight_decay=self.args.wd, nesterov=True)
        total_dist = torch.sum(self.client_dist, dim=0)
        virtual_representations = torch.zeros((int(torch.sum(total_dist).numpy()), self.mean.shape[-1]))
        virtual_labels = torch.zeros(int(torch.sum(total_dist).numpy())).long()
        cumsum = np.concatenate([[0], np.cumsum(total_dist.numpy())])

        for i in range(len(cumsum) - 1):
            dist = np.random.multivariate_normal(self.mean[i], self.var[i], size=(int(cumsum[i+1] - cumsum[i])))
            virtual_representations[int(cumsum[i]): int(cumsum[i+1])] = torch.tensor(dist)
            virtual_labels[int(cumsum[i]): int(cumsum[i+1])] = i
        # virtual_representations, virtual_labels, linear_rev = self.reverse_sampling()
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.args.crt_epoch):
            # logging.info(images.shape)
            images, labels = virtual_representations.to(self.device), virtual_labels.to(self.device)
            optimizer.zero_grad()
            probs = linear(images)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

        loss_func = torch.nn.L1Loss()

        self.AE.linear1.load_state_dict(self.model.clf.state_dict())
        virtual_representations = virtual_representations.to(self.device)
        self.AE.to(self.device)
        for epoch in range(self.args.crt_epoch * 100):
            # logging.info(images.shape)
            optimizer.zero_grad()
            output = self.AE(virtual_representations)
            loss = loss_func(output, virtual_representations)
            loss.backward()
            optimizer.step()
        print(loss.item())
        return linear