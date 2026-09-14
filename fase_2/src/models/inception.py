"""Individual PyTorch Inception reproduction; not the InceptionTime ensemble.

Architecture reference: https://github.com/hfawaz/InceptionTime
"""
import torch
from torch import nn


class InceptionModule(nn.Module):
    def __init__(self, channels, filters=32, bottleneck=32, kernel_size=40):
        super().__init__()
        self.reduce = nn.Conv1d(channels, bottleneck, 1, bias=False)
        self.branches = nn.ModuleList([nn.Conv1d(bottleneck, filters, kernel_size // 2**i,
            padding='same', bias=False) for i in range(3)])
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                  nn.Conv1d(channels, filters, 1, bias=False))
        self.norm = nn.BatchNorm1d(filters * 4)

    def forward(self, x):
        reduced = self.reduce(x)
        return torch.relu(self.norm(torch.cat([b(reduced) for b in self.branches]
                                             + [self.pool(x)], dim=1)))


class IndividualInceptionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, filters=32, depth=6, bottleneck=32, kernel_size=40):
        super().__init__()
        if depth < 3 or depth % 3:
            raise ValueError('depth must be a positive multiple of three')
        self.blocks, self.shortcuts = nn.ModuleList(), nn.ModuleList()
        channels = input_dim
        for _ in range(depth // 3):
            self.blocks.append(nn.Sequential(InceptionModule(channels, filters, bottleneck, kernel_size),
                InceptionModule(filters * 4, filters, bottleneck, kernel_size),
                InceptionModule(filters * 4, filters, bottleneck, kernel_size)))
            self.shortcuts.append(nn.Sequential(nn.Conv1d(channels, filters * 4, 1, bias=False),
                                                 nn.BatchNorm1d(filters * 4)))
            channels = filters * 4
        self.head = nn.Linear(channels, num_classes)

    def forward(self, values):
        x = values.transpose(1, 2)
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = torch.relu(block(x) + shortcut(x))
        return self.head(x.mean(dim=2))
