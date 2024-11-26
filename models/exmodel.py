import torch
from torch import nn
import torchvision.models as model

class DoNothing(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x;


class ExModel(nn.Module):
    def __init__(self, add_relu=True):
        super().__init__()

        self.resnet18 = model.resnet18(pretrained=True)
        self.classifier = nn.Linear(1000, 53)
    
        self.relu = (
            nn.ReLU() if add_relu
            else DoNothing()
                     )

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet18(x)
            x = self.relu(x)
        x = self.classifier(x)
        return x;
