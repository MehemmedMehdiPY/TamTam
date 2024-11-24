import torch
from processing import CardImageDataset

print(torch.__version__)
print(torch.cuda.is_available())

dataset = CardImageDataset(root='../dataset', mode='test')
image, label = dataset[0]

print(len(dataset))
print(image.shape, label.shape)