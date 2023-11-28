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
from sklearn.metrics import confusion_matrix


class CenterLoss(torch.nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, device=torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        torch.nn.init.kaiming_uniform_(self.centers, a=np.sqrt(5))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

in_features_dict = {
    'modVGG': 512,
    'SimpleCNN': 84,
    'resnet10': 512,
    'resnet18': 512,
    'resnet56': 2048
}

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True).to(self.device)
        self.clf_global = torch.clone(self.model.clf.weight.data.detach()).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,
                                         weight_decay=self.args.wd)

        self.center_loss = CenterLoss(self.num_classes, in_features_dict[self.args.net], self.device)
        self.centers = torch.zeros(self.num_classes).to(self.device)

    def get_dist(self):
        self.client_cnts = self.init_client_infos()
        dist = self.client_cnts / self.client_cnts.sum()  # 个数的比例

        return dist

    def load_client_state_dict(self, received_info):
        # If you want to customize how to state dict is loaded you can do so here
        # server_state_dict['clf.weight'] = torch.clone(self.model.state_dict()['clf.weight'])
        # server_state_dict['clf.omega'] = torch.clone(self.model.state_dict()['clf.omega'])
        # server_state_dict['clf.beta'] = torch.clone(self.model.state_dict()['clf.beta'])
        server_state_dict = received_info['global']
        self.model.load_state_dict(server_state_dict)

    def run(self, received_info):
        client_results = []
        for client_idx in self.client_map[self.round]:
            self.load_client_state_dict(received_info)
            # self.model.clf.margin = (received_info['distances'][client_idx] * self.args.mu).to(self.device)
            # self.centers = received_info['distances'][client_idx]
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.batch_size
            weights, centers = self.train()
            acc = self.test()
            dist = self.get_dist()
            client_results.append(
                {'weights': weights, 'num_samples': num_samples, 'acc': acc,
                 'client_index': self.client_index, 'dist': dist, 'centers': centers}
            )
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        # self.tsne_vis()
        return client_results

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        # self.centers = F.softmax(self.centers)
        # self.model.clf.margin = self.centers
        centers = None
        counts = torch.zeros(self.num_classes).to(self.device)
        epoch_loss = []

        hs = None
        labelss = None
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                h, probs = self.model(images, labels)

                # U, S, Vh = torch.linalg.svd(self.model.clf.weight.data, full_matrices=False)
                # U, Sg, Vh = torch.linalg.svd(self.clf_global, full_matrices=False)
                # phi = self.get_discriminability(h, labels)
                # output = F.linear(h, self.model.clf.weight)
                # g_output = F.linear(h, self.clf_global)
                loss = self.criterion(probs, labels) \
                    # + 0.3 * F.kl_div(F.log_softmax(output, dim=1),
                #                                                F.softmax(g_output, dim=1),
                #                                                reduction='batchmean')  # + 1 / phi
                loss.backward()

                self.optimizer.step()
                batch_loss.append(loss.item())
                # hs = h.detach() if hs is None else torch.cat([hs, h.detach().clone()], dim=0)
                # labelss = labels if labelss is None else torch.cat([labelss, labels.clone()], dim=0)
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                    epoch, sum(
                            epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        # self.get_discriminability(hs, labelss)

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
        self.model = self.model_type(self.num_classes, KD=True, margin=args.mu)
        wandb.watch(self.model)
        self.distances = {x: torch.zeros(self.num_classes).to(self.device) for x in range(self.args.client_number)}

    def start(self):
        with open('{}/config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [{'global': self.model.cpu().state_dict(), 'distances': torch.ones(self.num_classes).to(self.device)} for
                _ in range(self.args.thread_number)]

    def run(self, received_info):
        server_outputs = self.operations(received_info)
        acc = self.test()
        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:
            torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc
        for x in received_info:
            self.distances[x['client_index']] = x['centers']
        server_outputs = [{'global': g, 'distances': self.distances} for g in server_outputs]
        return server_outputs

    def operations(self, client_info):

        client_info.sort(key=lambda tup: tup['client_index'])
        clients_label_dist = [c['dist'] for c in client_info]
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]

        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])

        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        self.round += 1
        return [self.model.cpu().state_dict() for _ in range(self.args.thread_number)]

