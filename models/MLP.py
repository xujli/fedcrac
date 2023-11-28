"""The MLP model for PyTorch.

"""

import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    """The LeNet-5 model.

    Arguments:
        num_classes (int): The number of classes. Default: 10.
    """

    def __init__(self, num_classes=10, KD=False, projection=False):
        super().__init__()

        # We pad the image to get an input size of 32x32 as for the
        # original network in the LeCun paper
        self.fc1 = nn.Linear(784, 100)
        # self.bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, num_classes)
        self.KD = KD

    def flatten(self, x):
        """Flatten the tensor."""
        return x.view(x.size(0), -1)

    def forward(self, x):
        """Forward pass."""
        x = self.flatten(x)
        proj = self.fc1(x)
        x_f = F.relu(proj)
        # x = self.bn(x)
        x = self.fc2(x_f)

        if self.KD == True:
            return x_f, x
        else:
            return x

    def vis(self, x):
        x = self.flatten(x)
        proj = self.fc1(x)
        return proj
