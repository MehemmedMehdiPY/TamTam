import torch
from torch import nn
nn.CrossEntropyLoss

class DoNothing(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x;

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        torch.manual_seed(42)

        params = [(3, 16, 0, True), (16, 32, 0, True), (32, 64, 'same', True), (64, 32, 'same', False)]

        layers = []
        for in_channels, out_channels, padding, is_pool in params:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding)
            pool = (
                nn.MaxPool2d(kernel_size=3, stride=2) if is_pool 
                else DoNothing()
                )
            batch_norm = nn.BatchNorm2d(out_channels)
            relu = nn.ReLU()

            layers.append(nn.Sequential(conv, pool, batch_norm, relu))

        self.layers = layers
        self.backbone = nn.ModuleDict({'block_{}'.format(i): layer for i, layer in enumerate(layers)})
        
        self.flatten = nn.Flatten(start_dim=1)

        self.linear_1 = nn.Linear(26 * 26 * 32, 1024)
        self.batch_norm_1 = nn.BatchNorm1d(1024, affine=True)
        self.linear_2 = nn.Linear(1024, 128)
        self.linear_3 = nn.Linear(128, 53)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.backbone['block_0'](x)
        x = self.backbone['block_1'](x)
        x = self.backbone['block_2'](x)
        x = self.backbone['block_3'](x)
        x = self.flatten(x)

        x = self.relu(self.linear_1(x))
        x = self.batch_norm_1(x)
        x = self.relu(self.linear_2(x))
        x = self.linear_3(x)

        return x;

