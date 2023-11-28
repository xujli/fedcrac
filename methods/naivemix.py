'''
Naive implementation
'''

import torch
import wandb
import logging
import torch.nn.functional as F
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process
import numpy as np

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        self.global_data = client_dict['global_data']
        self.lam = self.args.gamma
        self._get_global_data_()

    def _get_global_data_(self):
        images_means, labels_means = torch.Tensor().to(self.device), torch.Tensor().to(self.device)
        for batch_idx, (images, labels) in enumerate(self.global_data):
            images, labels = images.to(self.device), labels.to(self.device)
            images_mean = torch.mean(images, dim=0).unsqueeze(0)
            labels_mean = torch.mean(F.one_hot(labels, num_classes=self.num_classes).float(), dim=0).unsqueeze(0)
            images_means = torch.cat([images_means, images_mean], dim=0)
            labels_means = torch.cat([labels_means, labels_mean], dim=0)

        self.images_means, self.label_means = images_means, labels_means

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
                batch_size = images.shape[0]
                num_2 = self.label_means.size()[0]
                idx2 = np.random.choice(range(num_2), batch_size, replace=False)
                images_2, labels_2 = self.images_means[idx2].to(self.device), self.label_means[idx2].to(self.device)
                # images_2 = images_2.repeat(batch_size, 1, 1, 1)
                # labels_2 = labels_2.repeat(batch_size, 1)

                inputX = (1 - self.lam) * images + self.lam * images_2
                log_probs = self.model(inputX)
                loss = self.mixup_criterion(log_probs, labels, labels_2, 1 - self.lam)

                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        return weights


    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
        
class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes)
        wandb.watch(self.model)
