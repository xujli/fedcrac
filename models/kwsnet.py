import torch.nn as nn
import torch.nn.functional as F
from models.MarginLinear import SoftmaxMargin


class M5(nn.Module):
    def __init__(self, n_classes: int = 20, KD=False, projection=False, margin=0):
        super().__init__()
        n_input, n_channel, stride = 80, 100, 32
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, 2 * n_channel)

        self.KD = KD
        self.projection = projection
        self.margin = margin != 0

        self.clf = (
            nn.Linear(2 * n_channel, n_classes)
            if not margin
            else SoftmaxMargin(2 * n_channel, n_classes, margin=margin)
        )

    def forward(self, x, target=None):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1).squeeze(1)
        x_f = self.fc1(x)

        X = self.clf(x_f) if not self.margin else self.clf(x_f, target)

        if self.KD == True:
            return x_f, X
        else:
            return X
