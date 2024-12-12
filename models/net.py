import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MarginLinear import (
    LSoftmaxLinear,
    SoftmaxMargin,
    DisAlignLinear,
    LGMLoss_v0,
    SoftmaxMarginMix,
)


class modVGG(nn.Module):
    def __init__(
        self, n_classes: int = 10, KD=False, projection=False, margin=0, mix=False
    ):
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
            nn.AdaptiveAvgPool2d((4, 4)),
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
        self.clf = (
            nn.Linear(512, n_classes)
            if not margin
            else Margin_method(512, n_classes, margin=margin)
        )
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


class modVGG2(nn.Module):
    def __init__(
        self, n_classes: int = 10, KD=False, projection=False, margin=0, mix=False
    ):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            # for tinyimagenet
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        if mix:
            Margin_method = SoftmaxMarginMix
        else:
            Margin_method = SoftmaxMargin
        self.clf = (
            nn.Linear(512, n_classes)
            if not margin
            else Margin_method(512, n_classes, margin=margin)
        )
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


class modVGG4(nn.Module):
    def __init__(
        self, n_classes: int = 10, KD=False, projection=False, margin=0, mix=False
    ):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.1),
            # for tinyimagenet
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.05),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        if mix:
            Margin_method = SoftmaxMarginMix
        else:
            Margin_method = SoftmaxMargin
        self.clf = (
            nn.Linear(512, n_classes)
            if not margin
            else Margin_method(512, n_classes, margin=margin)
        )
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


class modVGG3(nn.Module):
    def __init__(
        self, n_classes: int = 10, KD=False, projection=False, margin=0, mix=False
    ):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.1),
            # for tinyimagenet
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.1),
            # for imagenet
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        if mix:
            Margin_method = SoftmaxMarginMix
        else:
            Margin_method = SoftmaxMargin
        self.clf = (
            nn.Linear(512, n_classes)
            if not margin
            else Margin_method(512, n_classes, margin=margin)
        )
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


class modVGG3(nn.Module):
    def __init__(
        self, n_classes: int = 10, KD=False, projection=False, margin=0, mix=False
    ):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            # for tinyimagenet
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            # for imagenet
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        if mix:
            Margin_method = SoftmaxMarginMix
        else:
            Margin_method = SoftmaxMargin
        self.clf = (
            nn.Linear(512, n_classes)
            if not margin
            else Margin_method(512, n_classes, margin=margin)
        )
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


class modVGG6(nn.Module):
    def __init__(
        self, n_classes: int = 10, KD=False, projection=False, margin=0, mix=False
    ):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.15),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.15),
            # for tinyimagenet
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.15),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        if mix:
            Margin_method = SoftmaxMarginMix
        else:
            Margin_method = SoftmaxMargin
        self.clf = (
            nn.Linear(128, n_classes)
            if not margin
            else Margin_method(512, n_classes, margin=margin)
        )
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


class modVGG8(nn.Module):
    def __init__(
        self, n_classes: int = 10, KD=False, projection=False, margin=0, mix=False
    ):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.0),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.0),
            # for tinyimagenet
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.0),
            # for imagenet
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        if mix:
            Margin_method = SoftmaxMarginMix
        else:
            Margin_method = SoftmaxMargin
        self.clf = (
            nn.Linear(512, n_classes)
            if not margin
            else Margin_method(512, n_classes, margin=margin)
        )
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
        self.clf = (
            nn.Linear(84, n_classes)
            if not margin
            else SoftmaxMargin(84, n_classes, margin=margin)
        )
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

        self.clf = (
            nn.Linear(256, n_classes)
            if not margin
            else SoftmaxMargin(256, n_classes, margin=margin)
        )

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


import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast


# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]


model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        n_classes: int = 10,
        KD=False,
        projection=False,
        margin=0,
        mix=False,
        init_weights=True,
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
        )

        if mix:
            Margin_method = SoftmaxMarginMix
        else:
            Margin_method = SoftmaxMargin
        self.clf = (
            nn.Linear(1024, n_classes)
            if not margin
            else Margin_method(1024, n_classes, margin=margin)
        )
        self.KD = KD
        self.projection = projection
        self.margin = margin != 0

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor, target=None) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_f = self.classifier(x)
        if self.projection:
            x_p = self.p1(x_f)
            x_p = self.relu(x_p)
            x_f = self.p2(x_p) if not self.margin else self.clf(x_f, target)
        X = self.clf(x_f)

        if self.KD == True:
            return x_f, X
        else:
            return X

    def vis(self, X):
        X = self.features(X)
        return X

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # layers += [nn.Dropout(0.1)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(
    arch: str,
    cfg: str,
    batch_norm: bool,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13", "B", False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, **kwargs)


if __name__ == "__main__":
    model = vgg19_bn(n_classes=100, KD=True, margin=1)
    import pdb

    pdb.set_trace()
