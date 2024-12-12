import torch
import logging
import json
import wandb
import random
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from openTSNE import TSNE
from methods.utils import *
from sklearn.svm import LinearSVC
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from torch.multiprocessing import current_process


class Base_Client():
    def __init__(self, client_dict, args):
        self.train_data = client_dict['train_data']
        self.test_data = client_dict['test_data']
        self.device = 'cuda:{}'.format(client_dict['device'])
        self.model_type = client_dict['model_type']
        self.num_classes = client_dict['num_classes']
        self.args = args
        self.round = 0
        self.client_map = client_dict['client_map']
        self.train_dataloader = None
        self.test_dataloader = None
        self.client_index = None
        self.centers = None
        self.var = None

    def load_client_state_dict(self, server_state_dict):
        # If you want to customize how to state dict is loaded you can do so here
        self.model.load_state_dict(server_state_dict)

    def init_client_infos(self):
        client_cnts = torch.zeros(self.num_classes).float()
        for _, labels in self.train_dataloader:
            for label in labels.numpy():
                client_cnts[label] += 1
        return client_cnts

    def get_dist(self):
        self.client_cnts = self.init_client_infos()
        dist = self.client_cnts / self.client_cnts.sum()  # 个数的比例

        return dist

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
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device).long()
                self.optimizer.zero_grad()
                log_probs = self.model(images)
                if type(log_probs) == tuple:
                    hs, log_probs = log_probs
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                # logging.info(
                #     'client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                #                                                                                     epoch, sum(
                #             epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))

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

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        wandb_dict = {}
        test_correct = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_dataloader):
                x = x.to(self.device)
                target = target.to(self.device).long()

                pred = self.model(x)
                if type(pred) == tuple:
                    hs, pred = pred
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
            # logging.info(
            #     "************* Round {} Client {} Acc = {:.2f} **************".format(self.round, self.client_index,
            #                                                                           acc))

        return acc

    def personalized_test(self):
        self.model.to(self.device)
        self.model.eval()

        client_cnts = self.init_client_infos()
        test_sample_number = 0.0
        preds = torch.zeros(self.num_classes)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                pred = self.model(x)
                if type(pred) == tuple:
                    h, pred = pred
                _, predicted = torch.max(pred, 1)
                corrects = predicted.eq(target)
                for i in range(self.num_classes):
                    preds[i] = preds[i] + corrects[target == i].sum()

                test_sample_number += target.size(0)
            acc = (preds / client_cnts * (client_cnts / client_cnts.sum())) * 100
            # logging.info(
            #     "************* Round {} Client {} Per Acc = {:.2f} **************".format(self.round, self.client_index,
            #                                                                               acc))

        return acc

    def tsne_vis(self):
        self.model.to(self.device)
        self.model.eval()

        outputs = []
        labels = []
        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(self.test_dataloader):
                x = x.to(self.device)
                output = self.model.vis(x)
                outputs.extend(output.detach().cpu().numpy())
                labels.extend(label.numpy())

        outputs = np.reshape(np.array(outputs), (len(outputs), -1))

        tsne = TSNE(
            perplexity=30,
            metric="euclidean",
            n_jobs=8,
            random_state=42,
            verbose=True,
        )

        embedding = tsne.fit(outputs)
        fig, ax = plt.subplots()
        plot(embedding, labels, ax=ax)
        wandb.log({"client{}_tsne".format(self.client_index): wandb.Image(fig)})

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


class Base_Server():
    def __init__(self, server_dict, args):
        self.train_data = server_dict['train_data']
        self.test_data = server_dict['test_data']
        self.device = 'cuda:{}'.format(torch.cuda.device_count() - 1)
        self.model_type = server_dict['model_type']
        self.num_classes = server_dict['num_classes']
        self.acc = []
        self.round = 0
        self.args = args
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_path = server_dict['save_path']

    def run(self, received_info):
        server_outputs, clinet_acc = self.operations(received_info)
        acc = self.test(clinet_acc)
        self.log_info(received_info, acc)
        if self.acc and (acc > max(self.acc)):
            torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
        self.acc.append(acc)
        return server_outputs

    def start(self):
        with open('{}/config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [self.model.cpu().state_dict() for _ in range(self.args.thread_number)]

    def client_selection(self):
        mapping_dict = defaultdict(list)
        if self.args.client_sample < 1.0:
            num_clients = int(self.args.client_number * self.args.client_sample)
            client_list = random.sample(range(self.args.client_number), num_clients)
        else:
            num_clients = self.args.client_number
            client_list = list(range(num_clients))

        if num_clients % self.args.thread_number == 0 and num_clients > 0:
            clients_per_thread = int(num_clients / self.args.thread_number)
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t + clients_per_thread)]
                mapping_dict[c].append(idxs)

        return mapping_dict

    def log_info(self, client_info, acc):
        client_acc = sum([c['acc'] for c in client_info]) / len(client_info)
        out_str = 'Test/AccTop1: {}, Client_Train/AccTop1: {}, round: {}\n'.format(acc, client_acc, self.round)
        with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
            out_file.write(out_str)

    def _flatten_weights_from_model(self, model, device):
        """Return the weights of the given model as a 1-D tensor"""
        weights = torch.tensor([], requires_grad=False).to(device)
        model.to(device)
        for param in model.parameters():
            weights = torch.cat((weights, torch.flatten(param)))
        return weights

    def operations(self, client_info):

        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]

        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        client_acc = np.mean([c['acc'] for c in client_info])
        self.round += 1
        self.get_mean_var(client_info)
        return [self.model.cpu().state_dict() for _ in range(self.args.thread_number)], client_acc

    def get_mean_var(self, client_info):
        # calculate global mean and covariance
        self.client_dist = torch.stack([c['dist'] for c in client_info]).numpy()  # num_client, num_classes
        centers = np.stack([c['mean'] for c in client_info]) # num_client, num_classes, num_features
        self.mean = np.zeros(centers[0].shape)
        for c in range(self.mean.shape[0]):
            if np.sum(self.client_dist[:, c]) > 0:
                self.mean[c] = np.sum(centers[:, c] * np.expand_dims(self.client_dist[:, c], 1), axis=0) \
                               / np.sum(self.client_dist[:, c])
        
        self.mean = torch.from_numpy(self.mean).float()
        

        self.sim = torch.zeros((self.num_classes, self.num_classes))
        # inter class similarity
        for i in range(self.num_classes):
            self.sim[i, :] = F.cosine_similarity(self.mean[i], \
                self.mean, dim=1)

        self.dev = torch.zeros(self.num_classes)
        for i in range(self.num_classes):
            self.dev[i] += torch.sum(F.cosine_similarity(self.mean[i], torch.tensor(centers[:, i]), dim=1) * self.client_dist[:, i])
            self.dev[i] = self.dev[i] / np.sum(self.client_dist[:, i])

        
        cf = [c['weights']['clf.weight'] for c in client_info]
        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]
        mean_cf = sum([cf[i] * cw[i] for i in range(len(cf))])
        self.clf_dev = np.mean([torch.mean(torch.abs(cf[i] - mean_cf)) for i in range(len(cf))])


    def test(self, client_acc):
        self.model.to(self.device)
        self.model.eval()
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

                pred = self.model(x)
                if type(pred) == tuple:
                    h, pred = pred
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
            wandb_dict[self.args.method + "_dev".format(self.args.mu)] = torch.mean(self.dev)
            wandb_dict[self.args.method + "_clf_dev".format(self.args.mu)] = self.clf_dev
            wandb_dict[self.args.method + "_client_acc_gap".format(self.args.mu)] = acc - client_acc
            wandb_dict[self.args.method + "_phi_{}".format(self.args.mu)] = torch.sum(self.inter) / torch.sum(
                self.intra)
            if hasattr(self, 'sim'): 
                wandb_dict[self.args.method + "_sim_{}".format(self.args.mu)] = (torch.sum(self.sim) - len(self.sim)) / (self.sim.shape[0] * (self.sim.shape[1])) 

            wandb.log(wandb_dict)
            logging.info("************* Server Acc = {:.2f} **************".format(acc))

            if self.round == self.args.comm_round:
                logging.info("************* Max Server Acc = {:.2f}, Mean Acc of Last 10 Rounds = {:.2f} **************".format(max(self.acc), np.mean(self.acc[-10:])))
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

    def tsne_vis(self):
        self.model.to(self.device)
        self.model.eval()

        outputs = []
        labels = []
        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(self.test_data):
                x = x.to(self.device)
                output = self.model.vis(x)
                outputs.extend(output.detach().cpu().numpy())
                labels.extend(label.numpy())

        outputs = np.reshape(np.array(outputs), (len(outputs), -1))

        tsne = TSNE(
            perplexity=30,
            metric="euclidean",
            n_jobs=8,
            random_state=42,
            verbose=False,
        )

        embedding = tsne.fit(outputs)
        fig, ax = plt.subplots()
        plot(embedding, labels, ax=ax, title=self.args.method)
        wandb.log({"{}_server_tsne".format(self.args.method): wandb.Image(fig)})

    def decision_bound(self):
        self.model.to(self.device)
        self.model.eval()

        outputs = []
        labels = []
        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(self.test_data):
                x = x.to(self.device)
                output = self.model.vis(x)
                outputs.extend(output.detach().cpu().numpy())
                labels.extend(label.numpy())

        outputs = np.reshape(np.array(outputs), (len(outputs), -1))
        labels = np.array(labels)
        tsne = TSNE(
            perplexity=30,
            metric="euclidean",
            n_jobs=8,
            random_state=42,
            verbose=False,
        )

        steps = 1000
        embedding = tsne.fit(np.array(outputs))
        selected_embedding = embedding[(labels < 7) & (labels >= 4)] # >= labels[-3]
        selected_labels = labels[(labels < 7) & (labels >= 4)]

        xmin, xmax = np.min(embedding[:, 0]) - 1, np.max(embedding[:, 0]) + 1
        ymin, ymax = np.min(embedding[:, 1]) - 1, np.max(embedding[:, 1]) + 1

        x_span = np.linspace(xmin, xmax, steps)
        y_span = np.linspace(ymin, ymax, steps)
        xx, yy = np.meshgrid(x_span, y_span)
        clf = LinearSVC(max_iter=5e3)
        clf.fit(selected_embedding, selected_labels)
        predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = predictions.reshape(xx.shape)

        cmap = plt.get_cmap('jet')
        fig, ax = plt.subplots()
        ax.contourf(xx, yy, z, cmap=cmap, alpha=0.2)

        ax.scatter(selected_embedding[:, 0], selected_embedding[:, 1], c=selected_labels, cmap=cmap, lw=0, alpha=0.3)
        wandb.log({"{}_decision_boundary".format(self.args.method): wandb.Image(fig)})

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

    def finalize(self):
        self.tsne_vis()
        self.decision_bound()
