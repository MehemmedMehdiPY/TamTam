import numpy as np
from processing import CardImageDataset, RandomResizedCrop
from torchvision.transforms.v2 import RandomHorizontalFlip, Compose
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from models import SimpleClassifier
from train import Trainer

DEVICE = 'cpu'
BATCH_SIZE = 16
SEED = 42
EPOCHS = 2
NUM_CLASSES = 53

transforms = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomResizedCrop(p=0.5, size=(224, 224), scale=(0.8, 1.0), ratio=(3/4, 4/3))
])

train_dataset = CardImageDataset(root='../dataset', mode='train', transforms=transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = CardImageDataset(root='../dataset', mode='valid')
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SimpleClassifier()

for X, y in train_loader:
    out = model(X)
    break
print(out.shape)
print(out)
optimizer = Adam(params=model.parameters(), lr=0.001)
loss_fn = CrossEntropyLoss()

trainer = Trainer(model=model, train_loader=val_loader, val_loader=val_loader, optimizer=optimizer, 
        loss_fn=loss_fn, epochs=EPOCHS, filepath='./saved_models/trial_model.pt', num_classes=NUM_CLASSES, 
        device=DEVICE)
# trainer.save_model()