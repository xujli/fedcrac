import torch
import wandb
from methods.base import Base_Client, Base_Server

import logging
from torch.multiprocessing import current_process
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

in_features_dict = {
    'modVGG': 512,
    'SimpleCNN': 84,
    'resnet10': 512,
    'resnet18': 512,
    'resnet56': 2048
}


def match_loss(gw_syn, gw_real, device, dis_metric='mse'):
    dis = torch.tensor(0.0).to(device)

    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig].to(device)
            gws = gw_syn[ig].to(device)
            dis += distance_wb(gwr, gws)

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].to(device).reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].to(device).reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].to(device).reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].to(device).reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                    torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        # return 0

    dis_weight = torch.sum(
        1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,
                                         weight_decay=self.args.wd)
        self.eps = 1e-3
        self.centers = None
        self.cov = None

    def get_margin(self):
        self.client_cnts = self.init_client_infos()
        self.T = self.args.mu
        self.model.clf.margin = self.T / torch.pow(self.client_cnts, 1 / 4).to(self.device)

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
            weights, gradients = self.train()
            acc = self.test()
            client_results.append(
                {'weights': weights, 'num_samples': num_samples, 'acc': acc, 'client_index': self.client_index,
                 'dist': self.init_client_infos(), 'mean': self.centers, 'gradients': gradients})
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
                if self.centers is None:
                    self.centers = torch.zeros(self.num_classes, h.shape[1])
                    self.cov = torch.zeros(self.num_classes, h.shape[1], h.shape[1])
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
        features = {}
        self.model.eval()
        for batch_idx, (images, labels) in enumerate(self.train_dataloader):
            for label in torch.unique(labels):
                if int(label.numpy()) not in features.keys():
                    features[int(label.numpy())] = []
                features[int(label.numpy())].append(images[labels == label])


        gradients = {}
        for label, data in features.items():
            data = torch.cat(data).to(self.device)
            labels = torch.ones(data.shape[0]).to(self.device).long() * label

            h, log_probs = self.model(data)
            loss = self.criterion(log_probs, labels)

            grad = torch.autograd.grad(loss, list(self.model.clf.parameters()))
            grad = list((_.detach().data.cpu().clone() for _ in grad))
            gradients[label] = grad
            self.optimizer.zero_grad()

        weights = self.model.cpu().state_dict()
        return weights, gradients

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
        self.federated_features = torch.randn(size=(self.num_classes * 100, in_features_dict[args.net]),
                                              dtype=torch.float,
                                              requires_grad=True, device=self.device)
        self.federated_label = torch.tensor([np.ones(100) * i for i in range(self.num_classes)],
                                            dtype=torch.long, requires_grad=False, device=self.device).view(-1)
        self.feature_optimizer = torch.optim.SGD([self.federated_features, ], lr=args.lr, momentum=0.9, nesterov=True)

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
        self.client_dist = torch.stack([c['dist'] for c in client_info])  # num_client, num_classes
        gradients = [c['gradients'] for c in client_info]  # num_clients, num_classes, num_features

        logging.info("************* Gradient Matching **************")
        self.gradient_matching(gradients)
        logging.info("************* Classifier Retrain **************")
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

    def gradient_matching(self, gradients):
        linear = torch.nn.Linear(in_features_dict[self.args.net], self.num_classes).to(self.device)

        # ssd = {}
        # for name_param in reversed(self.model.state_dict()):
        #     if name_param == 'clf.bias':
        #         ssd['bias'] = torch.clone(self.model.state_dict()[name_param].cpu())
        #     if name_param == 'clf.weight':
        #         ssd['weight'] = torch.clone(self.model.state_dict()[name_param].cpu())
        #         break
        linear.load_state_dict(self.model.clf.state_dict())

        aggregated_gradients = {}
        for label in range(self.num_classes):
            avg_gradients = []
            for i, gradient in enumerate(gradients):
                if label in gradient.keys():
                    if len(avg_gradients) == 0:
                        avg_gradients.append(torch.clone(self.client_dist[i, label] * gradient[label][0]))
                        avg_gradients.append(torch.clone(self.client_dist[i, label] * gradient[label][1]))
                    else:
                        avg_gradients[0] += gradient[label][0] * self.client_dist[i, label]
                        avg_gradients[1] += gradient[label][1] * self.client_dist[i, label]
            if len(avg_gradients) > 0:
                avg_gradients[0] = (avg_gradients[0] / torch.sum(self.client_dist, dim=0)[label])
                avg_gradients[1] = (avg_gradients[1] / torch.sum(self.client_dist, dim=0)[label])

                aggregated_gradients[label] = avg_gradients

        for _ in range(self.args.match_epoch * 5):
            self.feature_optimizer.zero_grad()
            length = 100
            loss_feature = torch.tensor(0.0).to(self.device)
            for i in range(self.num_classes):
                if i in aggregated_gradients.keys():
                    features = self.federated_features[i * length: (i + 1) * length]
                    output = linear(features)
                    loss = self.criterion(output, self.federated_label[i * length: (i + 1) * length])
                    gw_syn = torch.autograd.grad(loss, list(linear.parameters()), create_graph=True)
                    loss_feature += match_loss(gw_syn, aggregated_gradients[i], self.device, )

            loss_feature.backward()
            self.feature_optimizer.step()

    def retrain(self):
        linear = torch.nn.Linear(in_features_dict[self.args.net], self.num_classes).to(self.device)
        optimizer = torch.optim.SGD(linear.parameters(), lr=self.args.lr, momentum=0.9,
                                    weight_decay=self.args.wd, nesterov=True)
        features = torch.clone(self.federated_features).detach().cpu()
        features.requires_grad = False

        dataset = TensorDataset(features, torch.clone(self.federated_label).cpu())
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.args.crt_epoch * 10):
            images, labels = features.to(self.device), self.federated_label.to(self.device)
            optimizer.zero_grad()
            probs = linear(images)
            loss = criterion(probs, labels)
            loss.backward()

            optimizer.step()

        return linear
