import torch
import logging
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process
import numpy as np
import wandb


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True, projection=False)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self.args.lr, momentum=0.9 if args.momentum else 0, weight_decay=self.args.wd)

        self.alpha = args.mu

    def init_client_infos(self):
        client_cnts = torch.zeros(self.num_classes).to(self.device).float()
        # print(self.client_infos)
        for _, labels in self.train_dataloader:
            for label in labels.numpy():
                client_cnts[label] += 1
        # print(client_cnts)
        return client_cnts

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
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]
            self.client_cnts = self.init_client_infos()
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.batch_size
            weights = self.train()
            acc = self.test()
            client_results.append(
                {'weights': weights, 'num_samples': num_samples, 'acc': acc, 'client_index': self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        return client_results

    def train(self):

        cidst = self.get_cdist()
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                hs, _ = self.model(images)
                ws = self.model.clf.weight

                logits = cidst * hs.mm(ws.transpose(0, 1))
                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                    epoch, sum(
                            epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))

        # 此处交换参数以及输出新字典
        # self.model.change_paras()
        weights = {key: value for key, value in self.model.cpu().state_dict().items()}
        return weights

    def test(self):

        if len(self.test_dataloader) == 0:
            return 0

        cidst = self.get_cdist()
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        # test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                hs, _ = self.model(x)
                ws = self.model.clf.weight

                logits = cidst * hs.mm(ws.transpose(0, 1))
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(logits, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item()
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number) * 100
            logging.info(
                "************* Round {} Client {} Acc = {:.2f} **************".format(self.round, self.client_index,
                                                                                      acc))
        return acc


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True, projection=False)
        wandb.watch(self.model)
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

