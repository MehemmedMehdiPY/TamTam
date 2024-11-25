import torch
from torch import nn


class DoNothing(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x;

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, is_pool=True):
        super().__init__()

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding)
        pool = (
            nn.MaxPool2d(kernel_size=3, stride=2) if is_pool
            else DoNothing()
            )
        batch_norm = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU()
        self.block = nn.Sequential(conv, pool, batch_norm, relu)

    def forward(self, x):
        return self.block(x);

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        torch.manual_seed(42)

        params = [(3, 16, 0, True), (16, 32, 0, True), (32, 64, 'same', True), (64, 32, 'same', False)]

        blocks = []
        for in_channels, out_channels, padding, is_pool in params:
            block = Block(in_channels=in_channels, out_channels=out_channels, padding=padding, is_pool=is_pool)
            blocks.append(block)
            
        self.backbone = nn.ModuleDict({'block_{}'.format(i): block for i, block in enumerate(blocks)})
        
        self.flatten = nn.Flatten(start_dim=1)

        self.linear_1 = nn.Linear(26 * 26 * 32, 1024)
        self.batch_norm_1 = nn.BatchNorm1d(1024, affine=True)
        self.linear_2 = nn.Linear(1024, 53)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.backbone['block_0'](x)
        x = self.backbone['block_1'](x)
        x = self.backbone['block_2'](x)
        x = self.backbone['block_3'](x)
        x = self.flatten(x)

        x = self.relu(self.linear_1(x))
        x = self.batch_norm_1(x)
        x = self.linear_2(x)

        return x;

