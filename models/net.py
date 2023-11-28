import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MarginLinear import LSoftmaxLinear, SoftmaxMargin, DisAlignLinear, LGMLoss_v0, SoftmaxMarginMix

class modVGG(nn.Module):
    def __init__(self, n_classes: int = 10, KD=False, projection=False, margin=0, mix=False):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        if mix:
            Margin_method = SoftmaxMarginMix
        else:
            Margin_method = SoftmaxMargin
        self.clf = nn.Linear(512, n_classes) if not margin else Margin_method(512, n_classes, margin=margin)
        self.KD = KD
        self.projection = projection
        self.margin = margin != 0
        self.relu = nn.ReLU()

        if projection:
            self.p1 = nn.Linear(512 * 1, 512 * 1)
            self.p2 = nn.Linear(512 * 1, 256)
            self.clf = nn.Linear(256, n_classes)

    def forward(self, X: torch.Tensor, target=None):
        x_f = self.seq(X)
        if self.projection:
            x_p = self.p1(x_f)
            x_p = self.relu(x_p)
            x_f = self.p2(x_p)
        X = self.clf(x_f) if not self.margin else self.clf(x_f, target)

        if self.KD == True:
            return x_f, X
        else:
            return X

    def vis(self, X):
        X = self.seq(X)
        return X


class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = 10, KD=False, projection=False, margin=0):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # 28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 10x10
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 5x5
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.clf = nn.Linear(84, n_classes) if not margin else SoftmaxMargin(84, n_classes, margin=margin)
        self.relu = nn.ReLU()

        self.KD = KD
        self.projection = projection
        self.margin = margin != 0

        if projection:
            self.p1 = nn.Linear(84 * 1, 84 * 1)
            self.p2 = nn.Linear(84 * 1, 256)
            self.clf = nn.Linear(256, n_classes)

    def forward(self, X: torch.Tensor, target=None):
        X = self.pool1(F.relu(self.conv1(X)))
        X = self.pool2(F.relu(self.conv2(X)))
        X = self.flatten(X)
        X = F.relu(self.fc1(X))
        x_f = F.relu(self.fc2(X))

        if self.projection:
            x_p = self.p1(x_f)
            x_p = self.relu(x_p)
            x_f = self.p2(x_p)
        X = self.clf(x_f) if not self.margin else self.clf(x_f, target)

        if self.KD == True:
            return x_f, X
        else:
            return X

    def vis(self, X):
        X = self.pool1(F.relu(self.conv1(X)))
        X = self.pool2(F.relu(self.conv2(X)))
        X = self.flatten(X)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        return X

class OneDCNN(nn.Module):
    def __init__(self, n_classes: int = 5, KD=False, projection=False, margin=0):
        super(OneDCNN, self).__init__()

        self.conv1 = nn.Conv1d(2, 32, 3)
        self.conv2 = nn.Conv1d(32, 32, 3)
        self.maxpool = nn.MaxPool1d(2, 2)

        self.conv3 = nn.Conv1d(32, 64, 3)
        self.conv4 = nn.Conv1d(64, 64, 3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(64 * 1, 256)
        self.linear2 = nn.Linear(256, 256)

        self.clf = nn.Linear(256, n_classes) if not margin else SoftmaxMargin(256, n_classes, margin=margin)

        self.relu = nn.ReLU()
        self.KD = KD
        self.projection = projection
        self.margin = margin != 0
        if projection:
            self.p1 = nn.Linear(256, 256)
            self.p2 = nn.Linear(256, 256)
            self.clf = nn.Linear(256, n_classes)


    def forward(self, x, target=None):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x_f = self.relu(self.linear1(x))

        if self.projection:
            x_p = self.p1(x_f)
            x_p = self.relu(x_p)
            x_f = self.p2(x_p)

        X = self.clf(x_f) if not self.margin else self.clf(x_f, target)

        if self.KD == True:
            return x_f, X
        else:
            return X

    def vis(self, X):
        x = self.relu(self.conv1(X))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        X = torch.flatten(x, 1)
        return X

if __name__ == '__main__':
    from thop import *
    model = SimpleCNN(10, margin=1, KD=True, projection=False)
    input = torch.randn(1, 3, 32, 32)
    # torchsummary.summary(model, (3, 32, 32))
    # print(output.shape)
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
