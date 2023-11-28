'''
Dataset Concstruction
Code based on https://github.com/FedML-AI/FedML
'''
import os
import torch
import logging
import numpy as np
import torch.utils.data as data
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import FashionMNIST, MNIST, EMNIST
from torchvision.datasets import DatasetFolder, ImageFolder
from natsort import natsorted

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
np.random.normal()
class CIFAR_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target, self.classes = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        if "cifar100" in self.root:
            cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)
        elif "cifar10" in self.root:
            cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target, cifar_dataobj.classes

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target, self.classes = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        if "fmnist" in self.root:
            mnist_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        elif "emnist" in self.root:
            mnist_dataobj = EMNIST(self.root, 'balanced',
                                   **{'train': self.train, 'transform': self.transform, 'target_transform': self.target_transform,
                                    'download': self.download})
        else:
            mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = mnist_dataobj.data
            target = np.array(mnist_dataobj.targets)
        else:
            data = mnist_dataobj.data
            target = np.array(mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target, mnist_dataobj.classes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index].float(), self.target[index]
        img = torch.unsqueeze(img, 0)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


# Imagenet
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.data = np.load(os.path.join(root, 'trainData.npy'), allow_pickle=True)
            self.target = np.load(os.path.join(root, 'trainLabel.npy'), allow_pickle=True)
        else:
            self.data = np.load(os.path.join(root, 'valData.npy'), allow_pickle=True)
            self.target = np.load(os.path.join(root, 'valLabel.npy'), allow_pickle=True)

        self.classes = np.sort(np.unique(self.target))

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.target[index]
        print(sample.shape)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.target)
        else:
            return len(self.dataidxs)

class ISCX_truncated(data.Dataset):
    def __init__(self, data_dir, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.data_dir = data_dir
        self.dataidxs = dataidxs
        self.train = train
        if self.train:
            self.data = np.load(os.path.join(data_dir, 'trainData.npy'), allow_pickle=True)
            self.target = np.load(os.path.join(data_dir, 'trainLabel.npy'), allow_pickle=True)
        else:
            self.data = np.load(os.path.join(data_dir, 'testData.npy'), allow_pickle=True)
            self.target = np.load(os.path.join(data_dir, 'testLabel.npy'), allow_pickle=True)

        self.data = np.concatenate([np.expand_dims(self.data, 1), np.expand_dims(self.data, 1), np.expand_dims(self.data, 1)], axis=1)
        self.target = self.target.astype(np.int)
        self.classes = np.unique(self.target)
        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.target = self.target[self.dataidxs]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = torch.tensor(self.data[index]).float(), torch.tensor(self.target[index]).long()

        return img, target


    def __len__(self):
        return len(self.data)

class QUIC_truncated(data.Dataset):

    def __init__(self, data_dir, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.data_dir = data_dir
        self.dataidxs = dataidxs
        self.train = train
        if self.train:
            self.data = np.load(os.path.join(data_dir, 'trainData.npy'), allow_pickle=True)
            self.target = np.load(os.path.join(data_dir, 'trainLabel.npy'), allow_pickle=True)
        else:
            valData = np.load(os.path.join(data_dir, 'valData.npy'), allow_pickle=True)
            valTarget = np.load(os.path.join(data_dir, 'valLabel.npy'), allow_pickle=True)
            testData = np.load(os.path.join(data_dir, 'testData.npy'), allow_pickle=True)
            testTarget = np.load(os.path.join(data_dir, 'testLabel.npy'), allow_pickle=True)
            self.data = np.concatenate([valData, testData])
            self.target = np.concatenate([valTarget, testTarget])
        print(len(self.data))
        self.data = np.reshape(self.data, (self.data.shape[0], -1, 2))
        self.data = np.transpose(self.data, [0, 2, 1])
        self.target = self.target[:, -1] - 1
        self.target = self.target.astype(np.int)
        self.classes = np.unique(self.target)
        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.target = self.target[self.dataidxs]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = torch.tensor(self.data[index]).float(), torch.tensor(self.target[index]).long()

        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from collections import Counter
    dataset = QUIC_truncated('../data/QUIC/')
    print(dataset.data.shape, Counter(dataset.target))
    dataset = QUIC_truncated('../data/QUIC/', train=False)