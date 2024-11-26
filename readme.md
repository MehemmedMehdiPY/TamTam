# Introduction
TamTam Project for Card Image Classification


## Installation

```
pip install -r requirements.txt
```

Please, note that I used CPU-supported PyTorch package. For your environment, you may wish to adjust CUDA version.

See [installation guideline](https://pytorch.org/get-started/locally/) for PyTorch CUDA.

## Usage
The below sample code requires the path to **data folder** and another path to **models to be saved** during training. Please, adjust your code based on your file hierarchy. During training, make sure you don't overwrite the trained models with the same filepaths.

```
import warnings
warnings.filterwarnings('ignore')
from TamTam.processing import CardImageDataset, RandomResizedCrop
from torchvision.transforms.v2 import RandomHorizontalFlip, Compose
from TamTam.models import SimpleClassifier, ExModel
from TamTam.train import Trainer
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


DEVICE = 'cpu' # or 'cuda'
BATCH_SIZE = 16
NUM_CLASSES = 53
LEARNING_RATE = 0.001
EPOCHS = 20 

# You may use different transforms
transforms = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomResizedCrop(p=0.5, size=(224, 224), scale=(0.8, 1.0), ratio=(3/4, 4/3))
])

train_dataset = CardImageDataset(root='./data', mode='train', transforms=transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = CardImageDataset(root='./data', mode='valid')
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = ExModel().to(DEVICE)

optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)
loss_fn = CrossEntropyLoss()

trainer = Trainer(model=model, train_loader=val_loader, val_loader=val_loader, optimizer=optimizer, 
        loss_fn=loss_fn, epochs=EPOCHS, filepath='./saved_models/trial_model.pt', num_classes=NUM_CLASSES, 
        device=DEVICE)
trainer.run()
```

You can either use SimpleClassifier with basic structure of Convolutional layers. Note that the model framework is prone to underfitting due to insufficient capacity.

```
model = SimpleClassifier().to(DEVICE)
```

## License
Free