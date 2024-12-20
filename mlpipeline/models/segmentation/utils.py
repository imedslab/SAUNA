import torch.nn as nn


def get_stages(self):
    return [
        nn.Sequential(self.conv1, self.bn1, self.relu),
        nn.Sequential(self.maxpool, self.layer1),
        self.layer2,
        self.layer3,
        self.layer4,
    ]
