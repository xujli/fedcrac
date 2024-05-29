import torch
import wandb
import copy
import json
import logging
import torch.nn as nn
import torch.functional as F
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def reg_ETF(output, label, classifier, mse_loss):
#    cur_M = classifier.cur_M
    target = classifier.cur_M[:, label].T  ## B, d
    loss = mse_loss(output, target)
    return loss

def dot_loss(output, label, cur_M, classifier, criterion, H_length, reg_lam=0):
    target = cur_M[:, label].T ## B, d  output: B, d
    if criterion == 'dot_loss':
        loss = - torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1).mean()
    elif criterion == 'reg_dot_loss':
        dot = torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1) #+ classifier.module.bias[label].view(-1)

        with torch.no_grad():
            M_length = torch.sqrt(torch.sum(target ** 2, dim=1, keepdims=False))
        loss = (1/2) * torch.mean(((dot-(M_length * H_length)) ** 2) / H_length)

        if reg_lam > 0:
            reg_Eh_l2 = torch.mean(torch.sqrt(torch.sum(output ** 2, dim=1, keepdims=True)))
            loss = loss + reg_Eh_l2*reg_lam

    return loss

def produce_Ew(label, num_classes):
    uni_label, count = torch.unique(label, return_counts=True)
    batch_size = label.size(0)
    uni_label_num = uni_label.size(0)
    assert batch_size == torch.sum(count)
    gamma = batch_size / uni_label_num
    Ew = torch.ones(1, num_classes).cuda(label.device)
    for i in range(uni_label_num):
        label_id = uni_label[i]
        label_count = count[i]
        length = torch.sqrt(gamma / label_count)
#        length = (gamma / label_count)
        #length = torch.sqrt(label_count / gamma)
        Ew[0, label_id] = length
    return Ew

def produce_global_Ew(cls_num_list):
    num_classes = len(cls_num_list)
    cls_num_list = torch.tensor(cls_num_list).cuda()
    total_num = torch.sum(cls_num_list)
    gamma = total_num / num_classes
    Ew = torch.sqrt(gamma / cls_num_list)
    Ew = Ew.unsqueeze(0)
    return Ew

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)




class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False):
        super(ETF_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M.cuda()

        self.LWS = LWS
        self.reg_ETF = reg_ETF

        self.BN_H = nn.BatchNorm1d(feat_in)
        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        if x.size(0) > 1:  # Only apply BatchNorm if batch size > 1
            x = self.BN_H(x)
            
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), min=1e-8)
        
        return x

    
    def gen_sparse_ETF(self, feat_in=512, num_classes=100, beta=0.6):
        # Initialize ETF
        etf = copy.deepcopy(self.ori_M)
        # Sparsify ETF
        num_zero_elements = int(beta * feat_in * num_classes)
        zero_indices = np.random.choice(feat_in * num_classes, num_zero_elements, replace=False)
        etf_flatten = etf.flatten()
        etf_flatten[zero_indices] = 0
        sparse_etf = etf_flatten.reshape(feat_in, num_classes)
        
        # Adjust non-zero elements
        sparse_etf = torch.tensor(sparse_etf, requires_grad=True)
        
        
        # Create a mask where the initial tensor is non-zero
        mask = (sparse_etf != 0).float()

        # Optimizer
        optimizer = torch.optim.Adam([sparse_etf], lr=0.0001)

        # Number of optimization steps
        n_steps = 10000
        loss_log = []
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Constraint 1: L2 norm of each row should be 1
            row_norms = torch.norm(sparse_etf, p=2, dim=0)
            norm_loss = torch.sum((row_norms - 0.1)**2)
            
            # Constraint 2: Maximize the angle between vectors (minimize cosine similarity)
            normalized_etf = sparse_etf / row_norms
            cos_sim = torch.mm(normalized_etf.t(), normalized_etf)
            torch.diagonal(cos_sim).fill_(-1)
            angle_loss = -torch.acos(cos_sim.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
            # angle_loss = -torch.sum(torch.acos(torch.clamp(cos_sim, -0.9999999, 0.9999999))) 
            # Total loss
            loss = norm_loss + angle_loss
    
            # Backpropagation
            loss.backward()
            
            # Apply the mask to the gradients
            if sparse_etf.grad is not None:
                sparse_etf.grad *= mask
                
                
            optimizer.step()
            loss_log.append(loss)
            if step % 100 == 0:
                print(f"Step {step}, Loss {loss.item()}")
                
        self.test_etf(sparse_etf)     
                
        return sparse_etf
    
    def test_etf(self, sparse_etf):
        # Normalize each column to have L2 norm = 1
        col_norms = torch.norm(sparse_etf, p=2, dim=0, keepdim=True)
        normalized_etf = sparse_etf / col_norms

        # Compute cosine similarities
        cosine_similarities = torch.mm(normalized_etf.t(), normalized_etf)

        # Zero out the diagonal (we don't want to compare vectors with themselves)
        torch.diagonal(cosine_similarities).fill_(float('nan'))

        # Compute angles in radians
        angles_radians = torch.acos(torch.clamp(cosine_similarities, -1, 1))

        # Convert angles from radians to degrees
        angles_degrees = angles_radians * (180 / np.pi)

        # Convert to numpy array
        angles_degrees_numpy = angles_degrees.cpu().detach().numpy()

        # Calculate mean and variance of angles, ignoring NaNs
        angle_mean = np.nanmean(angles_degrees_numpy)
        angle_variance = np.nanvar(angles_degrees_numpy)

        # Calculate mean and variance of norms
        col_norms_numpy = col_norms.cpu().detach().numpy()
        norm_mean = np.mean(col_norms_numpy)
        norm_variance = np.var(col_norms_numpy)

        print(f"Angle Mean: {angle_mean}, Angle Variance: {angle_variance}")
        print(f"Norm Mean: {norm_mean}, Norm Variance: {norm_variance}")



class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,#, nesterov=True,
                                         weight_decay=self.args.wd)

    def load_client_state_dict(self, server_state_dict):
        # If you want to customize how to state dict is loaded you can do so here
        server_state_dict, g_head, g_aux, l_heads = server_state_dict
        self.model.load_state_dict(server_state_dict)
        return g_head, g_aux, l_heads
    
    def run(self, received_info):
        client_results = []
        for client_idx in self.client_map[self.round]:
            g_head, g_aux, l_heads = self.load_client_state_dict(received_info)
            l_head = l_heads[client_idx]
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.batch_size
            if self.round >= self.args.comm_round:
                weights, g_aux, l_head = self.update_weights_norm_init(g_aux, l_head)
            else:
                weights, g_aux, l_head = self.train(g_head, g_aux, l_head)
            dist = self.init_client_infos()
            acc = self.test(g_head, l_head)
            client_results.append(
                {'weights': weights, 'num_samples': num_samples, 'acc': acc, 'client_index': self.client_index, 'dist': dist,
                 'mean': self.centers, 'cov': self.var, 'g_aux': g_aux.cpu().state_dict(), 'l_head': l_head.cpu().state_dict()})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        # self.tsne_vis()
        return client_results

    def train(self, g_head, g_aux, l_head):
        # def update_weights_gaux(self, net, , epoch, mu=1, lr=None, loss_switch=None):
        self.model.train()
        self.model.to(self.device)
        g_head.train()
        g_head.to(self.device)
        g_aux.train()
        g_aux.to(self.device)
        l_head.train()
        l_head.to(self.device)
        # train and update
        optimizer_g_backbone = torch.optim.SGD(list(self.model.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(g_aux.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l_head = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(
                    self.device), labels.to(self.device)

                labels = labels.long()
                optimizer_g_backbone.zero_grad()
                optimizer_g_aux.zero_grad()
                optimizer_l_head.zero_grad()
                # net.zero_grad()

                # backbone
                features, logits = self.model(images)

                output_g_backbone = g_head(features)
            
                loss_g_backbone = criterion_g(output_g_backbone, labels)
                loss_g_backbone.backward()
                
                optimizer_g_backbone.step()
                
                # g_aux
                output_g_aux = g_aux(features.detach())
                loss_g_aux = criterion_l(output_g_aux, labels)
                loss_g_aux.backward()
                optimizer_g_aux.step()

                # p_head
                output_l_head = l_head(features.detach())
                loss_l_head = criterion_l(output_l_head, labels)
                loss_l_head.backward()
                optimizer_l_head.step()

                loss = loss_g_backbone + loss_g_aux + loss_l_head
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
                
        return weights, g_aux, l_head

    def test(self, g_head, l_head):
        self.model.to(self.device)
        self.model.eval()
        g_head.eval()
        l_head.eval()
        p_mode = 1
        self.client_cnts = self.init_client_infos()

        if p_mode == 1:
            # 方案1：
            zero_classes = np.where(self.client_cnts == 0)[0]
            for i in zero_classes:
                g_head.weight.data[i, :] = -1e10
                l_head.weight.data[i, :] = -1e10

        wandb_dict = {}
        test_correct = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_dataloader):
                x = x.to(self.device)
                target = target.to(self.device).long()

                output = self.model(x)
                if type(output) == tuple:
                    hs, pred = output
                    pred = l_head(hs) + g_head(hs)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number) * 100
            wandb_dict[self.args.method + "_clinet:{}_acc".format(self.client_index)] = acc
            # wandb_dict[self.args.method + "_loss"] = loss
            # wandb.log(wandb_dict)
            logging.info(
                "************* Round {} Client {} Acc = {:.2f} **************".format(self.round, self.client_index,
                                                                                      acc))

        return acc

    ## gain personalzied l_head
    def update_weights_norm_init(self, g_aux, l_head):
        self.model.train()
        self.model.to(self.device)
        g_aux.train()
        g_aux.to(self.device)
        l_head.train()
        l_head.to(self.device)


        # 权重norm初始化
        norm = torch.norm(l_head.weight, p=2, dim=1)
        # 将g_head.weight转换为torch.nn.Parameter类型
        g_aux.weight = nn.Parameter(g_aux.weight * norm.unsqueeze(1))

        optimizer_g_aux = torch.optim.SGD(g_aux.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        criterion_ce = nn.CrossEntropyLoss()

        epoch_loss = []


        for epoch in range(self.args.epochs):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(
                    self.device), labels.to(self.device)

                labels = labels.long()
                optimizer_g_aux.zero_grad()

                features, _ = self.model(images, latent_output=True)
                
                outputs_g_aux = g_aux(features.detach())

                
                loss = criterion_ce(outputs_g_aux, labels)
                loss.backward()
                optimizer_g_aux.step()
                
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return self.model.cpu().state_dict(), copy.deepcopy(g_aux), copy.deepcopy(l_head)



class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True)
        wandb.watch(self.model)

        self.prepare()


    def prepare(self):
        in_features, out_features = self.model.clf.in_features, self.model.clf.out_features
        # 初始化ETF分类器 
        etf = ETF_Classifier(in_features, out_features) 
        # 新建线性层,权重使用ETF分类器的ori_M
        g_head = nn.Linear(in_features, out_features).to(self.device) 
        sparse_etf_mat = etf.gen_sparse_ETF(feat_in = in_features, num_classes = out_features, beta=0.6)
        g_head.weight.data = sparse_etf_mat.to(self.device)
        g_head.weight.data = g_head.weight.data.t()

        g_aux = nn.Linear(in_features, out_features).to(self.device)

        self.g_head = g_head
        self.g_aux = g_aux

        self.l_heads = []
        for i in range(self.args.client_number):
            self.l_heads.append(nn.Linear(in_features, out_features).to(self.device))

    def start(self):
        with open('{}/config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [[self.model.cpu().state_dict(), copy.deepcopy(self.g_head), copy.deepcopy(self.g_aux), copy.deepcopy(self.l_heads)] for _ in range(self.args.thread_number)]
    
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
        g_auxs_sd = [c['g_aux'] for c in client_info]
        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]

        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)
        ssd = self.g_head.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(g_auxs_sd)])
        self.g_head.load_state_dict(ssd)

        for client in client_info:
            self.l_heads[client['client_index']].load_state_dict(client['l_head'])

        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))

        client_acc = np.mean([c['acc'] for c in client_info])
        self.round += 1
        # self.get_mean_var(client_info)
        return [[self.model.cpu().state_dict(), copy.deepcopy(self.g_head), copy.deepcopy(self.g_aux), copy.deepcopy(self.l_heads)] for _ in range(self.args.thread_number)], client_acc

    def calibra(self):
        cali_alpha = torch.norm(self.g_aux.weight, dim=1)

        # 矫正feats
        # 计算 cali_alpha 的倒数
        cali_alpha = torch.pow(cali_alpha, 1)
        inverse_cali_alpha = 1.7 / cali_alpha
        # 将 inverse_cali_alpha 扩展为 (100, 1) 的形状
        inverse_cali_alpha = inverse_cali_alpha.view(-1, 1)

        # 矫正cls
        self.g_aux.weight = torch.nn.Parameter(self.g_aux.weight * inverse_cali_alpha)

    def test(self, client_acc, use_g_head=False):
        self.model.to(self.device)
        self.model.eval()
        self.g_head.eval()
        self.g_aux.eval()
        hs = None
        labelss = None
        preds = None
        logits = None

        wandb_dict = {}
        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_data):
                x = x.to(self.device)
                target = target.to(self.device).long()

                output = self.model(x)
                if type(output) == tuple:
                    h, _ = output
                    pred = self.g_head(h) if use_g_head else self.g_aux(h)

                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
                hs = h.detach() if hs is None else torch.cat([hs, h.detach().clone()], dim=0)
                labelss = target if labelss is None else torch.cat([labelss, target.clone()], dim=0)
                logits = pred.detach() if logits is None else torch.cat([logits, pred.detach().clone()], dim=0)
                preds = predicted.detach() if preds is None else torch.cat([preds, predicted.detach().clone()], dim=0)

            acc = (test_correct / test_sample_number) * 100
            loss = (test_loss / test_sample_number)
            self.get_discriminability(hs, labelss)
            wandb_dict[self.args.method + "_acc".format(self.args.mu)] = acc
            wandb_dict[self.args.method + "_loss".format(self.args.mu)] = loss
            # wandb_dict[self.args.method + "_dev".format(self.args.mu)] = torch.mean(self.dev)
            # wandb_dict[self.args.method + "_clf_dev".format(self.args.mu)] = self.clf_dev
            wandb_dict[self.args.method + "_client_acc_gap".format(self.args.mu)] = acc - client_acc
            wandb_dict[self.args.method + "_phi_{}".format(self.args.mu)] = torch.sum(self.inter) / torch.sum(
                self.intra)
            if hasattr(self, 'sim'): 
                wandb_dict[self.args.method + "_sim_{}".format(self.args.mu)] = (torch.sum(self.sim) - len(self.sim)) / (self.sim.shape[0] * (self.sim.shape[1])) 

            wandb.log(wandb_dict)
            logging.info("************* Server Acc = {:.2f} **************".format(acc))

            if self.round == self.args.comm_round:
                table = wandb.Table(data=pd.DataFrame(logits.cpu().numpy()))
                wandb.log({'{} logits'.format(self.args.method): table})
                
                matrix = confusion_matrix(labelss.cpu().numpy(), preds.cpu().numpy())
                acc_per_class = matrix.diagonal() / matrix.sum(axis=1)
                table = wandb.Table(
                    data=pd.DataFrame({'class': [i for i in range(self.num_classes)], 'accuracy': acc_per_class}))

                wandb.log(
                    {'{} accuracy for each class'.format(self.args.method): wandb.plot.bar(table, 'class', 'accuracy',
                                                                                           title='Acc for each class')})

                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
                theta = np.arange(0, 2 * np.pi, 2 * np.pi / self.num_classes)
                bar = ax.bar(theta, acc_per_class, alpha=0.5, width=2 * np.pi / self.num_classes)
                for r, bar in zip(acc_per_class, bar):
                    bar.set_facecolor(plt.cm.rainbow(r))

                wandb.log({"{} acc_per_class".format(self.args.method): wandb.Image(ax)})

                table = wandb.Table(data=pd.DataFrame({'{} prediction'.format(self.args.method): preds.cpu().numpy()}))
                wandb.log({'{} prediction'.format(self.args.method): table})

                table = wandb.Table(data=pd.DataFrame(
                    {'class': [i for i in range(self.num_classes)], 'phi': self.inter.detach().cpu().numpy()}))
                wandb.log({'{} inter discriminability'.format(self.args.method): wandb.plot.bar(table, 'class', 'phi',
                                                                                                title='Inter Discriminability'), })

                table = wandb.Table(data=pd.DataFrame(
                    {'class': [i for i in range(self.num_classes)], 'phi': self.intra.detach().cpu().numpy()}))
                wandb.log(
                    {'{} intra discriminability'.format(self.args.method): wandb.plot.bar(table, 'class', 'phi',
                                                                                          title='Intra Discriminability')})

                table = wandb.Table(data=pd.DataFrame(
                    {'intra': self.intra.detach().cpu().numpy(), 'inter': self.inter.detach().cpu().numpy()}))
                wandb.log({'{} intra / inter'.format(self.args.method): wandb.plot.scatter(table, 'intra', 'inter',
                                                                                           title='Intra / Inter')})
        return acc
