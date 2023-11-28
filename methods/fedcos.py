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
import math

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
        self.model = self.model_type(self.num_classes, KD=True, margin=args.mu).to(self.device)
        self.clf_global = torch.clone(self.model.clf.weight.data.detach()).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9,
                                         weight_decay=self.args.wd, nesterov=True)

    def get_dist(self):
        self.client_cnts = self.init_client_infos()
        dist = self.client_cnts / self.client_cnts.sum()  # 个数的比例

        return dist

    def run(self, received_info):
        client_results = []
        for client_idx in self.client_map[self.round]:
            self.load_client_state_dict(received_info)
            # self.model.clf.margin = (received_info['distances'][client_idx] * self.args.mu).to(self.device)
            self.centers = received_info
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.batch_size
            weights = self.train()
            acc = self.test()
            dist = self.get_dist()
            client_results.append(
                {'weights': weights, 'num_samples': num_samples, 'acc': acc,
                 'client_index': self.client_index, 'dist': dist}
            )
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        return client_results

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []

        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                h, probs = self.model(images, labels)

                correlation_loss = torch.sum(h * torch.index_select(self.model.clf.weight, dim=0, index=labels),
                                             dim=1) / (torch.norm(h, dim=1) * torch.norm(
                                       torch.index_select(self.model.clf.weight, dim=0, index=labels), dim=1))
                correlation_loss = torch.pow(correlation_loss - 1, 2)
                loss = self.criterion(probs, labels) + 0.2 * torch.mean(correlation_loss)
                loss.backward()

                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                    epoch, sum(
                            epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()

        return weights

    def test(self):
        self.model.to(self.device)
        self.model.eval()
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
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number) * 100
            wandb_dict[self.args.method + "_clinet:{}_acc".format(self.client_index)] = acc
            logging.info(
                "************* Round {} Client {} Acc = {:.2f} **************".format(self.round, self.client_index,
                                                                                      acc))
        return acc

    def get_discriminability(self, hs, labels):

        self.centers = torch.zeros((self.num_classes, hs.shape[1])).to(self.device)
        self.inter = torch.zeros(self.num_classes).to(self.device)
        self.intra = torch.zeros(self.num_classes).to(self.device)
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            selected_hs = hs[labels == label, :]
            self.centers[label] = torch.mean(selected_hs / (torch.norm(selected_hs, dim=1, keepdim=True)), dim=0)
            self.intra[label] = torch.mean(
                (selected_hs / torch.norm(selected_hs, dim=1, keepdim=True) - self.centers[label]) ** 2)

        for label in unique_labels:
            self.inter[label] = torch.mean((self.centers - self.centers[label]) ** 2) / (
                    self.num_classes - 1) * self.num_classes

        return torch.sum(self.inter) / torch.sum(
            self.intra)


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True, margin=args.mu)
        wandb.watch(self.model)
        self.centers = torch.zeros(self.num_classes, in_features_dict[self.args.net])

    def start(self):
        with open('{}/config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [self.model.cpu().state_dict() for \
                _ in range(self.args.thread_number)]

    def run(self, received_info):
        server_outputs = self.operations(received_info)
        acc = self.test()
        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:
            torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc
        return server_outputs

    def test(self):
        self.model.to(self.device)
        self.model.eval()
        hs = None
        labelss = None
        preds = None

        wandb_dict = {}
        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_data):
                x = x.to(self.device)
                target = target.to(self.device)

                h, pred = self.model(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
                hs = h.detach() if hs is None else torch.cat([hs, h.detach().clone()], dim=0)
                labelss = target if labelss is None else torch.cat([labelss, target.clone()], dim=0)
                preds = predicted.detach() if preds is None else torch.cat([preds, predicted.detach().clone()], dim=0)

            acc = (test_correct / test_sample_number) * 100
            loss = (test_loss / test_sample_number)
            self.get_discriminability(hs, labelss)
            wandb_dict[self.args.method + "_acc".format(self.args.mu)] = acc
            wandb_dict[self.args.method + "_loss".format(self.args.mu)] = loss
            wandb_dict[self.args.method + "_phi_{}".format(self.args.mu)] = torch.sum(self.inter) / torch.sum(
                self.intra)
            wandb.log(wandb_dict)
            if self.round == self.args.comm_round:
                matrix = confusion_matrix(labelss.cpu().numpy(), preds.cpu().numpy())
                acc_per_class = matrix.diagonal() / matrix.sum(axis=1)
                table = wandb.Table(
                    data=pd.DataFrame({'class': [i for i in range(self.num_classes)], 'accuracy': acc_per_class}))
                wandb.log({'accuracy for each class': wandb.plot.bar(table, 'class', 'accuracy',
                                                                     title='Acc for each class')})

                ax = plt.subplot(111, projection='polar')
                theta = np.arange(0, 2 * np.pi, 2 * np.pi / 10)
                bar = ax.bar(theta, acc_per_class, alpha=0.5, width=2 * np.pi / 10)
                for r, bar in zip(acc_per_class, bar):
                    bar.set_facecolor(plt.cm.rainbow(r))

                wandb.log({"acc_per_class": wandb.Image(ax)})

                table = wandb.Table(data=pd.DataFrame(
                    {'class': [i for i in range(self.num_classes)], 'phi': self.inter.detach().cpu().numpy()}))
                wandb.log({'inter discriminability': wandb.plot.bar(table, 'class', 'phi',
                                                                    title='Inter Discriminability')})

                table = wandb.Table(data=pd.DataFrame(
                    {'class': [i for i in range(self.num_classes)], 'phi': self.intra.detach().cpu().numpy()}))
                wandb.log(
                    {'intra discriminability': wandb.plot.bar(table, 'class', 'phi', title='Intra Discriminability')})
                table = wandb.Table(data=pd.DataFrame(
                    {'intra': self.intra.detach().cpu().numpy(), 'inter': self.inter.detach().cpu().numpy()}))
                wandb.log({'intra / inter': wandb.plot.scatter(table, 'intra', 'inter', title='Intra / Inter')})
            logging.info("************* Server Acc = {:.2f} **************".format(acc))
        return acc
