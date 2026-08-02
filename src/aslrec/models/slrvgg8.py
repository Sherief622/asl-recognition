"""SLRVGG-8: a VGG-style variant of SLRNet-8 designed for this project.

Seven 3x3 convolutions (vs SLRNet-8's six 5x5), BatchNorm after every conv,
512 filters in the deep layers (vs 384), global average pooling, and a
256-unit fully-connected head (vs 84): eight weight layers counting the
hidden FC, with the -8 in the name inherited from SLRNet-8's depth
designation. No explicit softmax: CrossEntropyLoss applies log-softmax
internally.

Layer names and ordering are identical to the original notebook so the
historical checkpoint (slrvgg8_best.pth) loads with strict=True.
"""

import torch.nn as nn


class SLRVGG8(nn.Module):
    def __init__(self, num_classes):
        super(SLRVGG8, self).__init__()
        self.features = nn.Sequential(
            # block 1: 224x224x32 -> 112x112x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # block 2: 112x112x64 -> 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # block 3: 56x56x128 -> 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # block 4: 28x28x256 -> 14x14x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # block 5: 14x14x256 -> 7x7x256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # blocks 6-7: 7x7x512, then global average pool to 1x1x512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
