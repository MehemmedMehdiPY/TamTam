import warnings
warnings.filterwarnings('ignore')
from processing import CardImageDataset, RandomResizedCrop
from torchvision.transforms.v2 import RandomHorizontalFlip, Compose
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from models import SimpleClassifier, ExModel
from train import Trainer

DEVICE = 'cpu'
BATCH_SIZE = 16
SEED = 42
NUM_CLASSES = 53
LEARNING_RATE = 0.001
EPOCHS = 20

# You may use different transforms
transforms = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomResizedCrop(p=0.5, size=(224, 224), scale=(0.8, 1.0), ratio=(3/4, 4/3))
])

train_dataset = CardImageDataset(root='../data', mode='train', transforms=transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = CardImageDataset(root='../data', mode='valid')
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SimpleClassifier().to(DEVICE)
# model = ExModel().to(DEVICE)

optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)
loss_fn = CrossEntropyLoss()

trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, 
        loss_fn=loss_fn, epochs=EPOCHS, filepath='./saved_models/trial_model.pt', num_classes=NUM_CLASSES, 
        device=DEVICE)
trainer.run()