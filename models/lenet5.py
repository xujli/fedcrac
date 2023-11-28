"""The LeNet-5 model for PyTorch.

Reference:

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to
document recognition." Proceedings of the IEEE, November 1998.
"""
import collections

import torch.nn as nn
import torch.nn.functional as F

from models.MarginLinear import LSoftmaxLinear, SoftmaxMargin

class Lenet5(nn.Module):
    """The LeNet-5 model.

    Arguments:
        num_classes (int): The number of classes. Default: 10.
    """

    def __init__(self, num_classes=10, in_channels=1, KD=False, projection=False, margin=0):
        super().__init__()

        # We pad the image to get an input size of 32x32 as for the
        # original network in the LeCun paper
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=6,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=0,
                               bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=120,
                               kernel_size=5,
                               bias=True)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(120 if in_channels == 1 else 480, 84)
        self.relu4 = nn.ReLU()
        self.clf = nn.Linear(84, num_classes) if margin==0 else SoftmaxMargin(84, num_classes, margin=margin)

        self.KD = KD
        self.projection = projection
        self.margin = margin != 0
        self.relu = nn.ReLU()

        if projection:
            self.p1 = nn.Linear(84 * 1, 84 * 1)
            self.p2 = nn.Linear(84 * 1, 256)
            self.clf = nn.Linear(256, num_classes)

        # Preparing named layers so that the model can be split and straddle
        # across the client and the server
        self.layers = []
        self.layerdict = collections.OrderedDict()
        self.layerdict['conv1'] = self.conv1
        self.layerdict['relu1'] = self.relu1
        self.layerdict['pool1'] = self.pool1
        self.layerdict['conv2'] = self.conv2
        self.layerdict['relu2'] = self.relu2
        self.layerdict['pool2'] = self.pool2
        self.layerdict['conv3'] = self.conv3
        self.layerdict['relu3'] = self.relu3
        self.layerdict['flatten'] = self.flatten
        self.layerdict['fc4'] = self.fc4
        self.layerdict['relu4'] = self.relu4
        self.layerdict['clf'] = self.clf
        self.layers.append('conv1')
        self.layers.append('relu1')
        self.layers.append('pool1')
        self.layers.append('conv2')
        self.layers.append('relu2')
        self.layers.append('pool2')
        self.layers.append('conv3')
        self.layers.append('relu3')
        self.layers.append('flatten')
        self.layers.append('fc4')
        self.layers.append('relu4')
        self.layers.append('clf')

    def flatten(self, x):
        """Flatten the tensor."""
        return x.view(x.size(0), -1)

    def forward(self, x, target=None):
        """Forward pass."""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc4(x)
        x_f = self.relu4(x)

        if self.projection:
            x_p = self.p1(x_f)
            x_p = self.relu(x_p)
            x_f = self.p2(x_p)
        X = self.clf(x_f, target) if (self.margin and self.training) else self.clf(x_f)

        if self.KD == True:
            return x_f, X
        else:
            return X

if __name__ == '__main__':

    model = Lenet5(10)
    print(84 * 84 / sum(param.numel() for param in model.parameters()))

