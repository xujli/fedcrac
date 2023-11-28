import torch
import wandb
from methods.base import Base_Client, Base_Server

import logging

from torch.multiprocessing import current_process
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
in_features_dict = {
    'Lenet5': 84,
    'OneDCNN': 256,
    'modVGG': 512,
    'SimpleCNN': 84,
    'resnet10': 512,
    'resnet18': 512,
    'resnet56': 2048
}

def torch_cov(input_vec:torch.tensor):
    x = input_vec - torch.mean(input_vec, dim=0)
    cov_matrix = torch.matmul(x.T, x) / (x.shape[0]-1)
    return cov_matrix

def diag(vector):
    mat = np.zeros((vector.shape[0], vector.shape[0]))
    for i in range(vector.shape[0]):
        mat[i, i] = vector[i]
    return mat

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0, #nesterov=True,
                                         weight_decay=self.args.wd)
        self.eps = 1e-3
        self.centers = None
        self.var = None

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

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        # self.get_margin()
        epoch_loss = []

        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                h, log_probs = self.model(images)
                # log_probs = log_probs / torch.norm(self.model.clf.weight.data.detach(), 1)
                if self.centers is None:
                    self.centers = np.zeros((self.num_classes, h.shape[1]))
                    self.var = np.zeros((self.num_classes, h.shape[1], h.shape[1]))
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

        for index in range(self.num_classes):
            if torch.sum((labelss == index)) > 1:
                selected_features = features[labelss == index]
                self.centers[index, :] = np.mean(selected_features.numpy(), axis=0)
                self.var[index, :, :] = np.cov(selected_features.numpy().T)
                # self.cov[index, :, :].add_(torch.eye(features.shape[1]) * 0.001)
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
        return [self.model.cpu().state_dict() for _ in range(self.args.thread_number)]

    def mixup_data(self, x, y, gamma=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if gamma > 0:
            lam = np.random.beta(gamma, gamma)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return torch.mean(lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b))

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

        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.args.crt_epoch):
            # logging.info(images.shape)
            images, labels = virtual_representations.to(self.device), virtual_labels.to(self.device)
            optimizer.zero_grad()
            images, targets_a, targets_b, lam = self.mixup_data(images, labels,
                                                   self.args.gamma)
            images, targets_a, targets_b = map(torch.autograd.Variable, (images,
                                                  targets_a, targets_b))
            probs = linear(images)
            loss = self.mixup_criterion(probs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

        return linear