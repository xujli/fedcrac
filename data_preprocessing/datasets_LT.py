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
from torchvision.datasets import FashionMNIST, MNIST, EMNIST, SVHN

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class MNIST_LT_truncated(MNIST):
    cls_num = 10

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False, imb_type='exp', imb_ratio=1):
        super(MNIST_LT_truncated, self).__init__(root, train, transform=transform, target_transform=target_transform, download=download)

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if self.train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_ratio)
            self.gen_imbalanced_data(img_num_list)
        self.targets = np.array(self.targets)
        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.target = np.array(self.targets[self.dataidxs])

        self.data = np.transpose(self.data, (0, 2, 3 ,1))

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.target = new_targets

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
        img = torch.stack([img, img, img], dim=0)
        return img, target

    def __len__(self):
        return len(self.data)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class CIFAR10_LT_truncated(CIFAR10):
    cls_num = 10

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False, imb_type='exp', imb_ratio=1):
        super(CIFAR10_LT_truncated, self).__init__(root, train, transform=transform, target_transform=target_transform, download=download)

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if self.train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_ratio)
            self.gen_imbalanced_data(img_num_list)
        else:
            self.target = np.array(self.targets)

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.target = np.array(self.target[self.dataidxs])

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.target = np.array(new_targets)

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

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class CIFAR100_LT_truncated(CIFAR10_LT_truncated):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    cls_num = 100
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


# Imagenet
class ImageFolder_LT_custom(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, imb_type='exp', imb_ratio=1.):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.cls_num = 200

        if self.train:
            self.data = np.load(os.path.join(root, 'trainData.npy'), allow_pickle=True)
            self.target = np.load(os.path.join(root, 'trainLabel.npy'), allow_pickle=True)
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_ratio)
            self.gen_imbalanced_data(img_num_list)
        else:
            self.data = np.load(os.path.join(root, 'valData.npy'), allow_pickle=True)
            self.target = np.load(os.path.join(root, 'valLabel.npy'), allow_pickle=True)

        self.classes = np.sort(np.unique(self.target))

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.target[index]
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

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.target, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.target = np.array(new_targets)


class ISCX_LT_truncated(data.Dataset):
    def __init__(self, data_dir, dataidxs=None, train=True, transform=None,
                 target_transform=None, download=False, imb_type='exp', imb_ratio=1):

        self.data_dir = data_dir
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.train:
            self.data = np.load(os.path.join(data_dir, 'trainData.npy'), allow_pickle=True)
            self.target = np.load(os.path.join(data_dir, 'trainLabel.npy'), allow_pickle=True)
            img_num_list = self.get_img_num_per_cls(len(np.unique(self.target)), imb_type, imb_ratio)
            self.gen_imbalanced_data(img_num_list)
        else:
            self.data = np.load(os.path.join(data_dir, 'testData.npy'), allow_pickle=True)
            self.target = np.load(os.path.join(data_dir, 'testLabel.npy'), allow_pickle=True)

        self.data = np.concatenate([np.expand_dims(self.data, -1), np.expand_dims(self.data, -1), np.expand_dims(self.data, -1)], axis=-1)

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
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.target)
        else:
            return len(self.dataidxs)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.target, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.target = np.array(new_targets)

class SVHN_LT_truncated(SVHN):
    cls_num = 10

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False, imb_type='exp', imb_ratio=1):
        super(SVHN_LT_truncated, self).__init__(root, split='train' if train else 'test', transform=transform, target_transform=target_transform, download=download)

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        if self.train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_ratio)
            self.gen_imbalanced_data(img_num_list)
        else:
            self.target = np.array(self.labels)

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.target = np.array(self.target[self.dataidxs])
            
        self.data = np.transpose(self.data, (0, 2, 3 ,1))

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * len(selec_idx))
        new_data = np.vstack(new_data)
        self.data = new_data
        self.target = np.array(new_targets)

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

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
class TinyImage_LT_truncated(data.Dataset):
    
    def __init__(self, data_dir, dataidxs=None, train=True, transform=None, target_transform=None, download=False, imb_type='exp', imb_ratio=1):

        self.data_dir = data_dir
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        if self.train:
            data = np.load(os.path.join(data_dir, 'tiny_imagenet_train.npz'), allow_pickle=True)
        else:
            data = np.load(os.path.join(data_dir, 'tiny_imagenet_val.npz'), allow_pickle=True)
        
        self.data, self.target = data['images'], data['labels']
        self.data = np.transpose(self.data, (0, 3, 1 ,2))
        self.classes = np.unique(self.target)
        if self.train:
            img_num_list = self.get_img_num_per_cls(len(np.unique(self.target)), imb_type, imb_ratio)
            self.gen_imbalanced_data(img_num_list)
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
        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)
    
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.target, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.target = np.array(new_targets)

class CalTech_LT_truncated(data.Dataset):
    def __init__(self, data_dir, dataidxs=None, train=True, transform=None,
                 target_transform=None, download=False, imb_type='exp', imb_ratio=1):

        self.data_dir = data_dir
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        if self.train:
            data = np.load(os.path.join(data_dir, 'tiny_imagenet_train.npz'), allow_pickle=True)
        else:
            data = np.load(os.path.join(data_dir, 'tiny_imagenet_val.npz'), allow_pickle=True)

        self.data, self.target = data['images'], data['labels']
        self.classes = np.unique(self.target)

        if self.train:
            img_num_list = self.get_img_num_per_cls(len(np.unique(self.target)), imb_type, imb_ratio)
            self.gen_imbalanced_data(img_num_list)
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
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img.float(), target.long()

    def __len__(self):
        if self.dataidxs is None:
            return len(self.target)
        else:
            return len(self.dataidxs)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.target, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.target = np.array(new_targets)


if __name__ == '__main__':
    import pandas as pd
    from collections import Counter
    from torch.utils.data import DataLoader
    dataset = CalTech_LT_truncated('../data/IC/caltech256', imb_ratio=.1, train=True)
    import pdb; pdb.set_trace()
    # print(dataset.data.shape, len(dataset.target), np.unique(dataset.target), Counter(dataset.target))
