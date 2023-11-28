import torch
import logging
import json
import wandb
import numpy as np
import torch.nn.functional as F
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process
from torch.utils.data.sampler import WeightedRandomSampler

class DistillKL(torch.nn.Module):
    """KL divergence for distillation"""
    def __init__(self, Temp, mode):
        super(DistillKL, self).__init__()
        self.T = Temp
        self.mode = mode

    def forward(self, y_s, y_t):
        outputs = torch.log_softmax(y_s/self.T, dim=1)
        labels = torch.softmax(y_t/self.T, dim=1)

        if self.mode == "kl":
            loss = F.kl_div(outputs, labels)
        elif self.mode == "ce":
            outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
            loss = -torch.mean(outputs, dim=0, keepdim=False)
        else:
            raise NotImplementedError()
        return loss
    
class CEKD_Loss(torch.nn.Module):
    # knowledge distlillation loss + CE Loss (or hinge loss)
    def __init__(self, Temp=2, lamda=1):
        super().__init__()
        self.lamda = lamda
        self.loss_cls = 'ce'
        self.loss_kd = 'kl'
        
        self.criterion_cls = torch.nn.CrossEntropyLoss()

        self.criterion_kd = DistillKL(Temp)

    def forward(self, logits, labels, feat=None, feat_teacher=None, classfier_weight=None):
        loss_cls = self.criterion_cls(logits, labels)

        if feat_teacher is not None and self.lamda != 0:
            loss_kd = self.criterion_kd(feat, feat_teacher)
            
            loss = loss_cls + self.lamda * loss_kd
        else:   
            loss = loss_cls
            loss_kd = torch.tensor(0)

        return loss, loss_cls, loss_kd
    

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss
    
class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, epsilon=0.01, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)
    
class LwF_Loss(torch.nn.Module):
    # knowledge distlillation loss + CE Loss (or hinge loss)
    def __init__(self, num_cls, device):
        super().__init__()
        self.lamda = 1
        self.loss_cls = 'ce'
        self.loss_kd = 'ce'
        self.num_cls = num_cls
        self.device = device
        
        if self.loss_cls == "ce":
            self.criterion_cls = torch.nn.CrossEntropyLoss()
        elif self.loss_cls == "sce": # smooth ce
            self.criterion_cls = LabelSmoothingCrossEntropy()
        else:
            raise NotImplementedError()

        self.criterion_kd = DistillKL(2, self.loss_kd)

    def forward(self, labels, teacher_pred, logits, logit_aug=None):
        if logit_aug is None:
            logit_aug = logits
        pos_cls = torch.unique(labels).tolist()
        neg_cls = list(set([*range(self.num_cls)]).difference(set(pos_cls)))
        transformed_labels = torch.tensor([pos_cls.index(i) for i in labels]).to(logits.device)
        # print(logit_aug[:, pos_cls].shape, transformed_labels.shape, labels.shape)
        loss_cls = self.criterion_cls(logit_aug[:, pos_cls], transformed_labels)
        loss_kd = self.criterion_kd(logits[:, neg_cls], teacher_pred[:, neg_cls])

        preds = torch.softmax(logits[:, pos_cls], dim=1)
        logs_preds = torch.log_softmax(logits[:, pos_cls], dim=1)
        loss_ent = torch.sum(-preds*logs_preds)
        # print(loss_cls, loss_kd, loss_ent)

        if torch.isnan(loss_cls):
            print(pos_cls)
            print(logits[:, pos_cls].shape, labels.shape, max(transformed_labels))
            print(loss_cls)

        # select according to classes
        loss = loss_cls + self.lamda * loss_kd - 0.002 * loss_ent
        return loss, loss_cls, loss_kd

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True).to(self.device)
        self.teacher = self.model_type(self.num_classes, KD=True).to(self.device)
        self.criterion = LwF_Loss(self.num_classes, self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,#, nesterov=True,
                                         weight_decay=self.args.wd)

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
    
    def get_aug(self):
        dist = self.get_dist()

        # probability for augmentation for every class
        max_num = max(dist)     
        prob = torch.tensor([1.0-i/max_num for i in dist])

        # obtain features and labels under eval mode
        feat_list, label_list = [], []
        
        for (imgs, labels) in self.train_dataloader: 
            with torch.no_grad():
                imgs = imgs.to(self.device)
                h, log_probs = self.teacher(imgs)
                feat_list.append(h.cpu())
                label_list.append(labels)

        feat_list = torch.cat(feat_list, 0)
        label_list = torch.cat(label_list, 0)
        # per-cls features
        feats_per_cls = [[] for i in range(self.num_classes)] 
        unique_labels = np.unique(label_list)
        for label in unique_labels:
            feats_per_cls[label] = feat_list[label_list == label]
        
        # calculate the variance
        per_cls_cov = np.zeros((self.num_classes, feat_list.shape[-1], feat_list.shape[-1]))
        for i, feats in enumerate(feats_per_cls):
            if len(feats) > 1:
                per_cls_cov[i] = np.cov(feats.numpy().T)

        cov = np.average(per_cls_cov, axis=0, weights=dist)  # covariance for feature dimension, shape: e.g., (128, 128)

        # pre-generate deviation
        divider = 500
        augs = torch.from_numpy(np.random.multivariate_normal(
            mean = np.zeros(cov.shape[0]), 
            cov = cov,  # covariance for feature dimension, shape: e.g., (128, 128)
            size = divider)).float().to(self.device)
        return prob, list(unique_labels), augs

    def get_balanced_dl(self):
        dist = self.get_dist()
        dist[dist == 0] = 1
        weight = torch.zeros(len(self.train_dataloader.dataset))
        targets = self.train_dataloader.dataset.target
        unique_labels = np.unique(targets)
        for label in unique_labels:
            weight[targets == label] = 1 / dist[label]
        weight[weight == torch.inf] = 500.
        weight[weight == torch.nan] = 1e-7
        sampler = WeightedRandomSampler(weight.type('torch.DoubleTensor'), len(weight))
        self.train_loader_balanced = torch.utils.data.DataLoader(self.train_dataloader.dataset, \
                                        batch_size=self.args.batch_size, sampler=sampler)

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        self.teacher.load_state_dict(self.model.state_dict())
        pointer, divider = 0, 500
        self.get_balanced_dl()
        prob, unique_labels, augs = self.get_aug()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader_balanced):
                images, labels = images.to(self.device), labels.to(self.device).long()
                self.optimizer.zero_grad()
                log_probs = self.model(images)
                if type(log_probs) == tuple:
                    hs, log_probs = log_probs
                
                rand_list = torch.rand(len(labels)) 
                mask = rand_list < prob[labels]
                degree = 1
                aug_num = sum(mask).item()
                feat_aug = hs.clone()
                if aug_num > 0: 
                    if pointer + aug_num >= divider:
                        pointer = 0
                    feat_aug[mask] = feat_aug[mask] + augs[pointer: pointer + aug_num] * degree
                    pointer = pointer + aug_num  

                logits_aug = self.model.clf(feat_aug)

                with torch.no_grad():
                    _, pred_teacher = self.teacher(images)

                loss, loss_cls, loss_kd = self.criterion(labels, pred_teacher, log_probs, logits_aug)
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    'client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                    epoch, sum(
                            epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))

        # estimate
        features = None
        labelss = None
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                h, log_probs = self.model(images)
                if self.centers is None:
                    self.centers = np.zeros((self.num_classes, h.shape[1]))
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
                # self.var[index, :] = np.var(selected_features.numpy(), axis=0)
                # self.cov[index, :, :].add_(torch.eye(features.shape[1]) * 0.001)
        weights = self.model.cpu().state_dict()
        return weights

class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True)
        wandb.watch(self.model)
