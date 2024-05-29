import torch
import wandb
import copy
import numpy as np
import torch.nn.functional as F
from methods.base import Base_Client, Base_Server

import torch.nn as nn
from torch import zeros, ones, cat, add, mul, no_grad, eq, sigmoid
from torch.nn.functional import softmax, log_softmax

import data_preprocessing.data_loader as dl

class Ensemble_highway(nn.Module):

    def __init__(self, in_feature: int = 256, num_classes: int = 10):
        super(Ensemble_highway, self).__init__()
        # calibration
        self.ensemble_scale = nn.Parameter(ones(num_classes, 1))
        self.ensemble_bias = nn.Parameter(zeros(1))

        self.logit_scale = nn.Parameter(ones(num_classes))
        self.logit_bias = nn.Parameter(zeros(num_classes))
        self.classifier2 = nn.Linear(in_features=in_feature, out_features=1)
        self.carry_values = []
        self.weight_values = []

    def forward(self, clients_feature, clients_logit, new_logit):
        all_logits_weight = torch.mm(clients_logit[0], self.ensemble_scale)
        all_logits_weight = all_logits_weight + self.ensemble_bias
        all_logits_weight_sigmoid = sigmoid(all_logits_weight)
        for one_logit in clients_logit[1:]:
            new_value = torch.mm(one_logit, self.ensemble_scale)
            new_value = new_value + self.ensemble_bias
            new_value_sigmoid = sigmoid(new_value)
            all_logits_weight_sigmoid = cat((all_logits_weight_sigmoid, new_value_sigmoid), dim=1)
        norm1 = all_logits_weight_sigmoid.norm(1, dim=1)
        norm1 = norm1.unsqueeze(1).expand_as(all_logits_weight_sigmoid)
        all_logits_weight_norm = all_logits_weight_sigmoid / norm1
        all_logits_weight_norm = all_logits_weight_norm.t()
        weighted_logits = sum([
            one_weight.view(-1, 1) * one_logit
            for one_logit, one_weight in zip(clients_logit, all_logits_weight_norm)
        ]
        )
        avg_weight = ([1.0 / 8] * 8)
        weighted_feature = sum(
            [
                one_weight * one_feature
                for one_feature, one_weight in zip(clients_feature, avg_weight)
            ]
        )
        calibration_logit = weighted_logits * self.logit_scale + self.logit_bias
        carry_gate = self.classifier2(weighted_feature)
        carry_gate_sigmoid = sigmoid(carry_gate)
        finally_logit = carry_gate_sigmoid * calibration_logit + (1 - carry_gate_sigmoid) * new_logit
        return finally_logit

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,#, nesterov=True,
                                         weight_decay=self.args.wd)

in_features_dict = {
    'Lenet5': 84,
    'OneDCNN': 256,
    'SimpleCNN': 84,
    'modVGG': 512,
    'modVGG2': 512,
    'resnet10': 512,
    'resnet18': 512,
    'resnet56': 2048
}

class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True).to(self.device)
        self.model1 = self.model_type(self.num_classes, KD=True).to(self.device)
        self.model2 = self.model_type(self.num_classes, KD=True).to(self.device)
        wandb.watch(self.model)

        self.highway_model = Ensemble_highway(in_feature=in_features_dict[self.args.net])
        self.highway_model.to(self.device)
        self.dict_global_params = self.model.state_dict()
        self.dataset_global_teaching = server_dict['client_train_data'][args.client_number]
        self.total_steps = 100
        self.mini_batch_size = 20
        self.mini_batch_size_unlabled = 128
        self.ce_loss = nn.CrossEntropyLoss()
        self.lr_global_teaching = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_global_teaching, weight_decay=0.0002)
        self.highway_optimizer = torch.optim.Adam(self.highway_model.parameters(), lr=self.lr_global_teaching)
        self.fedavg_optimizer = torch.optim.Adam(self.model2.parameters(), lr=self.lr_global_teaching, weight_decay=0.0002)
        self.temperature = 2
        self.epoch_acc = []
        self.epoch_acc_eval = []
        self.epoch_loss = []
        self.epoch_avg_ensemble_acc = []
        self.init_fedavg_acc = []
        self.init_ensemble_acc = []
        self.disalign_ensemble_acc = []
        self.disalign_ensemble_eval_acc = []
        self.epoch_acc_fed_min = []
        self.epoch_acc_fed_max = []
        self.random_state = np.random.RandomState(1)
        self.server_steps = 200
        self.ld = 0
        self.ensemble_ld = 0.0
        self.unlabeled_data = server_dict['client_train_data'][args.client_number + 1]      


    def run(self, received_info):
        server_outputs, clinet_acc = self.operations(received_info)
        acc = self.test(clinet_acc)
        self.log_info(received_info, acc)
        if acc > self.acc:
            torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc
        return server_outputs 

    def operations(self, client_info):

        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]

        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])

        self.dict_global_params = ssd
        # self.update_distillation_highway_feature(client_sd)
        self.model.load_state_dict(self.dict_global_params)

        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))

        client_acc = np.mean([c['acc'] for c in client_info])
        self.round += 1
        # self.get_mean_var(client_info)
        return [self.model.cpu().state_dict() for _ in range(self.args.thread_number)], client_acc

    def update_distillation_highway_feature(self, list_dicts_local_params: list):
        self.model.to(self.device)

        self.model2.load_state_dict(self.dict_global_params)
        self.model2.train()
        cnt = 0
        for batch in self.dataset_global_teaching:
            images, labels = batch
            _, fedavg_outputs = self.model2(images.to(self.device))
            fedavg_hard_loss = self.ce_loss(fedavg_outputs, labels.to(self.device))
            self.fedavg_optimizer.zero_grad()
            fedavg_hard_loss.backward()
            self.fedavg_optimizer.step()
            
            cnt += 1
            if cnt >= self.server_steps:
                break

        self.model2.eval()
        self.highway_model.train()
            
        # cnt = 0
        # for batch in self.dataset_global_teaching:
        #     images, labels = batch
        #     images = images.to(self.device)
        #     labels = labels.to(self.device)

        #     ensemble_feature_temp, ensemble_logit_temp = self.features_logits(images, copy.deepcopy(
        #         list_dicts_local_params))
        #     _, fedavg_new_logits = self.model2(images)
        #     ensemble_avg_logit_finally = self.highway_model(ensemble_feature_temp,
        #                                                     ensemble_logit_temp, fedavg_new_logits)
        #     ensemble_hard_loss = self.ce_loss(ensemble_avg_logit_finally, labels)
        #     self.highway_optimizer.zero_grad()
        #     ensemble_hard_loss.backward()
        #     self.highway_optimizer.step()

        #     cnt += 1
        #     if cnt >= self.server_steps:
        #         break

        # self.highway_model.eval()
        self.model.load_state_dict(self.dict_global_params)
        self.model.train()
        cnt = 0
        for label_batch, unlabel_batch in zip(self.dataset_global_teaching, self.unlabeled_data):
            images_labeled, labels_train = label_batch[0].to(self.device), label_batch[1].to(self.device)
            images_unlabeled = unlabel_batch[0].to(self.device)

            # teacher_feature_temp, teacher_logits_temp = self.features_logits(images_unlabeled,
            #                                                                  copy.deepcopy(
            #                                                                      list_dicts_local_params))
            _, fedavg_unlabeled_logits = self.model2(images_unlabeled)
            # logits_teacher = self.highway_model(teacher_feature_temp, teacher_logits_temp,
            #                                     fedavg_unlabeled_logits)
            _, logits_student = self.model(images_unlabeled)
            x = log_softmax(logits_student / self.temperature, dim=1)
            y = softmax(fedavg_unlabeled_logits / self.temperature, dim=1)
            soft_loss = F.kl_div(x, y.detach(), reduction='batchmean')
            _, logits_student_train = self.model(images_labeled)
            hard_loss = self.ce_loss(logits_student_train, labels_train)
            total_loss = add(mul(soft_loss, self.ld), mul(hard_loss, 1 - self.ld))
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            cnt += 1
            if cnt >= self.server_steps:
                break

        self.dict_global_params = self.model.state_dict()

    def features_logits(self, images, list_dicts_local_params):
        list_features = []
        list_logits = []
        for dict_local_params in list_dicts_local_params:
            self.model1.load_state_dict(dict_local_params)
            self.model1.eval()
            with no_grad():
                local_feature, local_logits = self.model1(images)
                list_features.append(copy.deepcopy(local_feature))
                list_logits.append(copy.deepcopy(local_logits))
        return list_features, list_logits

    def eval(self, data_test, batch_size_test: int):
        self.model.load_state_dict(self.dict_global_params)
        self.model.eval()
        with no_grad():
            test_loader = torch.data.DataLoader(data_test, batch_size_test)
            num_corrects = 0

            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return copy.deepcopy(self.dict_global_params)
