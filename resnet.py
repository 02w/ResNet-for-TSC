import torch.nn as nn
import torch.nn.functional as F


def same_conv1d(in_channels, n_feature_maps, kernel_size):
    if kernel_size % 2 == 1:
        return nn.Conv1d(in_channels, n_feature_maps, kernel_size, padding=kernel_size // 2)
    else:
        return nn.Sequential(
            nn.ConstantPad1d((kernel_size // 2 - 1, kernel_size // 2), 0),
            nn.Conv1d(in_channels, n_feature_maps, kernel_size=kernel_size)
        )


def block(in_channels, n_feature_maps, kernel):
    return nn.Sequential(
        same_conv1d(in_channels, n_feature_maps, kernel[0]),
        nn.BatchNorm1d(n_feature_maps),
        nn.ReLU(inplace=True),

        same_conv1d(n_feature_maps, n_feature_maps, kernel[1]),
        nn.BatchNorm1d(n_feature_maps),
        nn.ReLU(inplace=True),

        same_conv1d(n_feature_maps, n_feature_maps, kernel[2]),
        nn.BatchNorm1d(n_feature_maps)
    )


def shortcut(in_channels, n_feature_maps):
    return nn.Sequential(
        nn.Conv1d(in_channels, n_feature_maps, kernel_size=1),
        nn.BatchNorm1d(n_feature_maps)
    )


class ResNet(nn.Module):
    def __init__(self, in_channels, n_feature_maps, n_classes, kernel_size: list):
        super().__init__()
        self.in_channels = in_channels
        self.n_feature_maps = n_feature_maps
        self.n_classes = n_classes

        assert len(kernel_size) == 3
        self.kernel_size = kernel_size

        self.conv1 = block(self.in_channels, self.n_feature_maps, self.kernel_size)
        self.shortcut1 = shortcut(self.in_channels, self.n_feature_maps)
        self.activation1 = nn.ReLU(inplace=True)

        self.conv2 = block(self.n_feature_maps, 2 * self.n_feature_maps, self.kernel_size)
        self.shortcut2 = shortcut(self.n_feature_maps, 2 * self.n_feature_maps)
        self.activation2 = nn.ReLU(inplace=True)

        self.conv3 = block(2 * self.n_feature_maps, 2 * self.n_feature_maps, self.kernel_size)
        self.shortcut3 = nn.BatchNorm1d(2 * self.n_feature_maps)
        self.activation3 = nn.ReLU(inplace=True)

        # global avg pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(2 * self.n_feature_maps, n_classes)

    def forward(self, x):
        conv = self.activation1(self.conv1(x) + self.shortcut1(x))
        conv = self.activation2(self.conv2(conv) + self.shortcut2(conv))
        conv = self.activation3(self.conv3(conv) + self.shortcut3(conv))

        output = self.gap(conv)
        output = self.fc(output.squeeze(2))
        output = F.log_softmax(output, dim=-1)
        return output
