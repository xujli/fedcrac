import torch
import wandb
import numpy as np
from methods.base import Base_Client, Base_Server

import logging

from torch.multiprocessing import current_process


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True, margin=1).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,
                                         weight_decay=self.args.wd)

    def get_margin(self):
        self.client_cnts = self.init_client_infos()
        self.T = self.args.mu
        margin = self.T / torch.pow(self.client_cnts, 1/4).to(self.device)
        margin = torch.where(torch.isinf(margin), torch.full_like(margin, 0), margin)
        self.model.clf.margin = margin

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
            dist = self.init_client_infos()
            client_results.append(
                {'weights': weights, 'num_samples': num_samples, 'acc': acc, 'client_index': self.client_index, 'dist': dist,
                 'mean': self.centers, 'cov': self.var})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        # self.tsne_vis()
        return client_results

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        self.get_margin()
        epoch_loss = []

        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                h, log_probs = self.model(images, labels)
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
                if self.centers is None:
                    self.centers = np.zeros((self.num_classes, h.shape[1]))

        for index in range(self.num_classes):
            if torch.sum((labelss == index)) > 1:
                selected_features = features[labelss == index]
                self.centers[index, :] = np.mean(selected_features.numpy(), axis=0)
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
        self.model = self.model_type(self.num_classes, KD=True, margin=1)
        wandb.watch(self.model)

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


