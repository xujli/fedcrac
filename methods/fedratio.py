import torch
import wandb
import json
import numpy as np
from copy import deepcopy
from methods.base import Base_Client, Base_Server



class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,#, nesterov=True,
                                         weight_decay=self.args.wd)
    

    def load_client_state_dict(self, received_info):
        server_state_dict = received_info[0]
        self.model.load_state_dict(server_state_dict)
        self.loss_weight = torch.clone(received_info[1]).to(self.device)

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
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.loss_weight.float()).to(self.device)
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

class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True)
        wandb.watch(self.model)
        self.alpha = 1.
        self.beta = 0.1
        n_aux = 64
        self.aux_data = self.get_aux_data(n_aux)
    
    def get_aux_data(self, n_aux) -> dict:
        aux_data = {i: [] for i in range(self.num_classes)}
        flgs = [0] * self.num_classes
        for data, labels in self.train_data:
            for i, label in enumerate(labels):
                label = label.item()
                if len(aux_data[label]) < n_aux:
                    aux_data[label].append(data[i])
                else:
                    if flgs[label] == 0:
                        flgs[label] = 1
                        aux_data[label] = torch.stack(aux_data[label])
            
            if sum(flgs) == self.num_classes:
                break
        return aux_data
    
    def start(self):
        with open('{}/config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [[self.model.cpu().state_dict(), torch.ones(self.num_classes)] for _ in range(self.args.thread_number)]
    
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
        
        cc_net = self.compute_cc()
        pos = self.outlier_detect(ssd, cc_net)
        loss_weight = self.whole_determination(pos, ssd, cc_net)
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        client_acc = np.mean([c['acc'] for c in client_info])
        self.round += 1
        self.get_mean_var(client_info)
        
        return [[self.model.cpu().state_dict(), loss_weight] for _ in range(self.args.thread_number)], client_acc
    
    def outlier_detect(self, w_global, w_locals):
        w_global = w_global['clf.weight'].cpu().numpy()
        w = []
        for i in range(len(w_locals)):
            temp = (w_locals[i]['clf.weight'].cpu().numpy() - w_global) * 100
            w.append(temp)
        res = self.search_neuron_new(w)
        return res

    def search_neuron_new(self, w):
        w = np.array(w)
        pos_res = np.zeros((len(w), self.num_classes, 512))
        for i in range(w.shape[1]):
            for j in range(w.shape[2]):
                temp = []
                for p in range(len(w)):
                    temp.append(w[p, i, j])
                max_index = temp.index(max(temp))
                # pos_res[max_index, i, j] = 1 

                if w[max_index, i, j] == 0:
                    outlier = np.where(temp == w[max_index, i, j])
                else:
                    outlier = np.where(np.abs(temp) / abs(w[max_index, i, j]) > 0.80)
                if len(outlier[0]) < 2:
                    pos_res[max_index, i, j] = 1
                # pos_res[max_index, i, j] = 1
        return pos_res
    
    def whole_determination(self, pos, w_glob_last, cc_net):
        ratio_res = []
        for it in range(self.num_classes):
            cc_class = it
            aux_sum = 0
            aux_other_sum = 0
            layer = 1
            for i in range(pos.shape[1]):
                for j in range(pos.shape[2]):
                    if pos[cc_class, i, j] == 1:
                        temp = []
                        last = w_glob_last['clf.weight'.format(layer)].cpu().numpy()[i, j]
                        cc = cc_net[cc_class]['clf.weight'.format(layer)].cpu().numpy()[i, j]
                        for p in range(len(cc_net)):
                            temp.append(cc_net[p]['clf.weight'.format(layer)].cpu().numpy()[i, j] - last)
                        temp = np.array(temp)
                        temp = np.delete(temp, cc_class)
                        temp_ave = np.sum(temp) / (len(cc_net) - 1)
                        aux_sum += cc - last
                        aux_other_sum += temp_ave
                        
            if aux_other_sum != 0:
                res = abs(aux_sum) / abs(aux_other_sum)
            else:
                res = 10
            print('label {}-----aux_data:{}, aux_other:{}, ratio:{}'.format(it, aux_sum, aux_other_sum, res))
            ratio_res.append(res)

        # normalize the radio alpha
        ratio_min = np.min(ratio_res)
        ratio_max = np.max(ratio_res)
        for i in range(len(ratio_res)):
            # add a upper bound to the ratio
            if ratio_res[i] >= 5000:
                ratio_res[i] = 5000
            ratio_res[i] = 1.0 + 0.1 * ratio_res[i]
            # ratio_res[i] = 1.5 - 0.3  * (ratio_res[i] - ratio_min) / (ratio_max - ratio_min)
        return torch.tensor(ratio_res)
    
    def compute_cc(self):
        self.model.to(self.device)
        ssd = deepcopy(self.model.state_dict())
        
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        Delta_W = []
        
        for c in range(self.num_classes):
            for _ in range(self.args.epochs):
                opt.zero_grad()
                data = self.aux_data[c]
                data, label = data.to(self.device), torch.tensor([c] * len(data)).to(self.device)
                pred = self.model(data)
                if type(pred) == tuple:
                    h, pred = pred
                loss = self.criterion(pred, label)
                loss.backward()
                opt.step()
            
            Delta_W.append(deepcopy(self.model.state_dict()))
            self.model.load_state_dict(ssd)
            
        self.model.cpu()
        return Delta_W

       