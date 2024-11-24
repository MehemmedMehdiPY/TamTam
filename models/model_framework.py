import torch
from torch import nn
import torchvision.models as model

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

        params = [(3, 64, 0, False), (64, 64, 0, True), (64, 128, 0, False), (128, 128, 0, True), 
                  (128, 256, 0, False), (256, 256, 'same', True), (256, 512, 'same', False)]

        blocks = []
        for in_channels, out_channels, padding, is_pool in params:
            block = Block(in_channels=in_channels, out_channels=out_channels, padding=padding, is_pool=is_pool)
            blocks.append(block)
        self.backbone = nn.ModuleDict({'block_{}'.format(i): block for i, block in enumerate(blocks)})
        self.global_pool = nn.AvgPool2d(kernel_size=24)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear_1 = nn.Linear(512, 128)
        self.linear_2 = nn.Linear(128, 53)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.backbone['block_0'](x)
        x = self.backbone['block_1'](x)
        x = self.backbone['block_2'](x)
        x = self.backbone['block_3'](x)
        x = self.backbone['block_4'](x)
        x = self.backbone['block_5'](x)
        x = self.backbone['block_6'](x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x;

class ExModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet18 = model.resnet18(pretrained=True)
        self.classifier = nn.Linear(1000, 53)

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet18(x)
        x = self.classifier(x)
        return x;
