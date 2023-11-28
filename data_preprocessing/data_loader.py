'''
Federated Dataset Loading and Partitioning
Code based on https://github.com/FedML-AI/FedML
'''
import os
import logging

import numpy as np
from numpy.core.fromnumeric import mean
import torch.utils.data as data
import torchvision.transforms as transforms

from scipy import io
from data_preprocessing.datasets import CIFAR_truncated, MNIST_truncated, QUIC_truncated, ISCX_truncated, ImageFolder_custom
from data_preprocessing.datasets_LT import CIFAR10_LT_truncated, CIFAR100_LT_truncated, MNIST_LT_truncated, ImageFolder_LT_custom, ISCX_LT_truncated
from data_preprocessing.samplers import iid, dirichlet, orthogonal

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def _data_transforms_cifar(datadir):
    if "cifar100" in datadir:
        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]
    else:
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def _data_transforms_mnist():
    train_transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    valid_transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return train_transform, valid_transform


def _data_transforms_emnist():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10,
                                translate=(0.2, 0.2),
                                scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    valid_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return train_transform, valid_transform


def _data_transforms_imagenet(datadir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop_scale = 0.08
    jitter_param = 0.4
    image_size = 64
    image_resize = 68

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size, scale=(crop_scale, 1.0)),
        transforms.ColorJitter(
            brightness=jitter_param, contrast=jitter_param,
            saturation=jitter_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, valid_transform

def _data_transforms_ISCX():
    # if "cifar100" in datadir:
    #     CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    #     CIFAR_STD = [0.2673, 0.2564, 0.2762]
    # else:
    #     CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    #     CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return train_transform, valid_transform

def load_data(datadir):
    if 'cifar' in datadir:
        train_transform, test_transform = _data_transforms_cifar(datadir)
        dl_obj = CIFAR_truncated
    elif 'mnist' in datadir:
        if 'emnist' in datadir:
            train_transform, test_transform = _data_transforms_emnist()
        else:
            train_transform, test_transform = _data_transforms_mnist()
        dl_obj = MNIST_truncated
    elif 'QUIC' in datadir:
        train_transform, test_transform = None, None
        dl_obj = QUIC_truncated
    elif 'ISCX' in datadir:
        train_transform, test_transform = _data_transforms_ISCX()
        dl_obj = ISCX_truncated
    else:
        train_transform, test_transform = _data_transforms_imagenet(datadir)
        dl_obj = ImageFolder_custom
    train_ds = dl_obj(datadir, train=True, download=True, transform=train_transform)
    test_ds = dl_obj(datadir, train=False, download=True, transform=test_transform)

    y_train, y_test = train_ds.target, test_ds.target

    return (y_train, y_test)


def partition_data(datadir, partition, n_nets, alpha, silos=5):
    logging.info("*********partition data***************")
    y_train, y_test = load_data(datadir)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    class_num = len(np.unique(y_train))

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = class_num
        N = n_train
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == 'orthogonal':
        net_dataidx_map = {}
        data_num = n_train // silos
        classes_silos = np.arange(0, class_num)
        for i in range(silos):
            if i != silos - 1:
                classes_ids = \
                    classes_silos[len(classes_silos) // silos * i: len(classes_silos) // silos * (1 + i)]
            else:
                classes_ids = \
                    classes_silos[len(classes_silos) // silos * i: -1]
            idx_i = []
            for class_id in classes_ids:
                idx_k = np.where(y_train == class_id)[0]
                idx_i.extend(idx_k)
            np.random.shuffle(idx_i)
            batch_idxs = np.array_split(idx_i, n_nets // silos)
            for j in range(n_nets // silos):
                net_dataidx_map[i * n_nets // silos + j] = batch_idxs[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return class_num, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(datadir, train_bs, test_bs, dataidxs=None):
    if 'cifar' in datadir:
        train_transform, test_transform = _data_transforms_cifar(datadir)
        dl_obj = CIFAR_truncated
    elif 'mnist' in datadir:
        train_transform, test_transform = _data_transforms_emnist() if 'emnist' in datadir else _data_transforms_mnist()
        dl_obj = MNIST_truncated
    elif 'QUIC' in datadir:
        train_transform, test_transform = None, None
        dl_obj = QUIC_truncated
    elif 'ISCX' in datadir:
        train_transform, test_transform = _data_transforms_ISCX()
        dl_obj = ISCX_truncated
    else:
        train_transform, test_transform = _data_transforms_imagenet(datadir)
        dl_obj = ImageFolder_custom
        
    workers = 0
    persist = False

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=train_transform, download=True)
    test_ds = dl_obj(datadir, train=False, transform=test_transform, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True, num_workers=workers,
                               persistent_workers=persist)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True, num_workers=workers,
                              persistent_workers=persist)

    return train_dl, test_dl


def load_partition_data(data_dir, partition_method, partition_alpha, client_number, batch_size, silos):
    # if not os.path.exists(
    #         'processed_{}_{}_{}_{}_{}'.format(data_dir.split('/')[1], partition_method, partition_alpha, client_number,
    #                                           batch_size)):
    class_num, net_dataidx_map, traindata_cls_counts = partition_data(data_dir, partition_method, client_number,
                                                                      partition_alpha, silos)

    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(data_dir, 5, 5)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(data_dir, batch_size, batch_size, dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num



def partition_data_LT(datadir, partition, n_nets, alpha, silos=5, imb_type='exp', imb_ratio=0.1):
    logging.info("*********partition data***************")
    y_train, y_test = load_data_LT(datadir, imb_type, imb_ratio)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    class_num = len(np.unique(y_train))

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = class_num
        N = n_train
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == 'orthogonal':
        net_dataidx_map = {}
        data_num = n_train // silos
        classes_silos = np.arange(0, class_num)
        for i in range(silos):
            if i != silos - 1:
                classes_ids = \
                    classes_silos[len(classes_silos) // silos * i: len(classes_silos) // silos * (1 + i)]
            else:
                classes_ids = \
                    classes_silos[len(classes_silos) // silos * i: -1]
            idx_i = []
            for class_id in classes_ids:
                idx_k = np.where(y_train == class_id)[0]
                idx_i.extend(idx_k)
            np.random.shuffle(idx_i)
            batch_idxs = np.array_split(idx_i, n_nets // silos)
            for j in range(n_nets // silos):
                net_dataidx_map[i * n_nets // silos + j] = batch_idxs[j]


    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return class_num, net_dataidx_map, traindata_cls_counts


def load_data_LT(datadir, imb_type, imb_ratio):
    if 'cifar100' in datadir:
        train_transform, test_transform = _data_transforms_cifar(datadir)
        dl_obj = CIFAR100_LT_truncated
    elif 'cifar10' in datadir:
        train_transform, test_transform = _data_transforms_cifar(datadir)
        dl_obj = CIFAR10_LT_truncated
    elif 'mnist' in datadir:
        if 'emnist' in datadir:
            train_transform, test_transform = _data_transforms_emnist()
        else:
            train_transform, test_transform = _data_transforms_mnist()
        dl_obj = MNIST_LT_truncated
    elif 'ISCX' in datadir:
        train_transform, test_transform = _data_transforms_ISCX()
        dl_obj = ISCX_LT_truncated
    else:
        train_transform, test_transform = _data_transforms_imagenet(datadir)
        dl_obj = ImageFolder_LT_custom
    train_ds = dl_obj(datadir, train=True, download=True, transform=train_transform, imb_type=imb_type, imb_ratio=imb_ratio)
    test_ds = dl_obj(datadir, train=False, download=True, transform=test_transform, imb_type=imb_type, imb_ratio=imb_ratio)

    y_train, y_test = train_ds.target, test_ds.target

    return (y_train, y_test)


# for centralized training
def get_dataloader_LT(datadir, train_bs, test_bs, dataidxs=None, imb_type='exp', imb_ratio=0.1):

    if 'cifar100' in datadir:
        train_transform, test_transform = _data_transforms_cifar(datadir)
        dl_obj = CIFAR100_LT_truncated
        workers = 0
        persist = False
    elif 'cifar10' in datadir:
        train_transform, test_transform = _data_transforms_cifar(datadir)
        dl_obj = CIFAR10_LT_truncated
        workers = 0
        persist = False
    elif 'mnist' in datadir:
        train_transform, test_transform = _data_transforms_emnist() if 'emnist' in datadir else _data_transforms_mnist()
        dl_obj = MNIST_LT_truncated
        workers = 0
        persist = False
    elif 'ISCX' in datadir:
        train_transform, test_transform = _data_transforms_ISCX()
        dl_obj = ISCX_LT_truncated
        workers = 0
        persist = False
    else:
        train_transform, test_transform = _data_transforms_imagenet(datadir)
        dl_obj = ImageFolder_LT_custom
        workers = 0
        persist = False

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=train_transform, download=True, imb_type=imb_type, imb_ratio=imb_ratio)
    test_ds = dl_obj(datadir, train=False, transform=test_transform, download=True, imb_type=imb_type, imb_ratio=imb_ratio)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True, num_workers=workers,
                               persistent_workers=persist)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True, num_workers=workers,
                              persistent_workers=persist)
    print(len(train_ds))
    return train_dl, test_dl

def load_partition_data_LT(data_dir, partition_method, partition_alpha, client_number, batch_size, silos, imb_type='exp', imb_ratio=0.5):
    class_num, net_dataidx_map, traindata_cls_counts = partition_data_LT(data_dir, partition_method, client_number,
                                                                      partition_alpha, silos, imb_type=imb_type, imb_ratio=imb_ratio)

    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader_LT(data_dir, 5, 5, imb_type=imb_type, imb_ratio=imb_ratio)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader_LT(data_dir, batch_size, batch_size, dataidxs, imb_type=imb_type, imb_ratio=imb_ratio)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

if __name__ == '__main__':
    import numpy as np
    from collections import Counter
    np.random.seed(1)
    client_number = 50

    class_num, net_dataidx_map, traindata_cls_counts = partition_data('../data/ISCX_1', 'hetero', client_number,
                                                                      1.0, 5)
    print(traindata_cls_counts)
    client_num_matrix = np.zeros((client_number, class_num))
    for client_idx in range(client_number):
        dataidxs = traindata_cls_counts[client_idx]
        for j in dataidxs.keys():
            client_num_matrix[client_idx, j] = dataidxs[j]

    import numpy as np
    import matplotlib.pyplot as plt
    plt.rc('font', family='Times New Roman')

    def im_hist(mat, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        im = ax.imshow(mat, cmap=plt.cm.get_cmap('Blues'), aspect='auto')

        # bins = np.arange(-lim, lim + binwidth, binwidth)
        x = np.sum(mat, axis=0)
        y = np.sum(mat, axis=1)
        ax_histx.bar(np.arange(len(x)), x)
        ax_histy.barh(np.arange(len(y)), y)
        return im

    def scatter_hist(x, y, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y)

        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax / binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    # rect_scatter = [left, bottom, width, height]
    # rect_histx = [left, bottom + height + spacing, width, 0.2]
    # rect_histy = [left + width + spacing, bottom, 0.2, height]

    fig = plt.figure(figsize=(40, class_num))

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2, width_ratios=(8, 2), height_ratios=(2, 8),
                          left=0.1, right=0.99, bottom=0.1, top=0.95,
                          wspace=0.01, hspace=0.04)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_histx.set_yticks(np.arange(0, int(np.max(np.sum(client_num_matrix, axis=1))), 500),
                        np.arange(0, int(np.max(np.sum(client_num_matrix, axis=1))), 500), fontsize=20)
    ax_histy.set_xticks(np.arange(0, int(np.max(np.sum(client_num_matrix, axis=0))), 2000),
                        np.arange(0, int(np.max(np.sum(client_num_matrix, axis=0))), 2000), fontsize=20)
    ax_histx.xaxis.set_visible(False)
    ax_histy.yaxis.set_visible(False)

    im = im_hist(client_num_matrix.T, ax, ax_histx, ax_histy)
    fontsize = 30
    ax.set_xlabel('Client ID', fontsize=30)
    ax.set_ylabel('Class ID', fontsize=30)
    ax.set_xticks(np.arange(0, 50, 10), np.arange(0, 50, 10), fontsize=30)
    ax.set_yticks(np.arange(0, class_num, 2), np.arange(0, class_num, 2), fontsize=30)
    # use the previously defined function

    ax_colorbar = fig.add_subplot(gs[0, 1])
    fig.colorbar(im, cax=ax_colorbar, orientation='horizontal')
    ax_colorbar.xaxis.set_ticks(np.arange(0, np.max(client_num_matrix), 100).astype(np.int),
                                np.arange(0, np.max(client_num_matrix), 100).astype(np.int), fontsize=30)
    ax_colorbar.xaxis.tick_top()
    # cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8])  # 设置colarbar位置
    # cbar = fig.colorbar(ax)  # 共享colorbar
    # plt.tight_layout(pad=.01)
    # plt.savefig()
    plt.show()