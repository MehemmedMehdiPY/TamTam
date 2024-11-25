from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from processing import RandomResizedCrop
from torchvision.transforms.v2 import RandomHorizontalFlip, Compose

transforms = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomResizedCrop(p=0.5, size=(224, 224), scale=(0.8, 1.0), ratio=(3/4, 4/3))
])

image = Image.open('../dataset/test/king of clubs/4.jpg')
image = np.asarray(image)
image = torch.tensor(image)

image = image.permute([2, 0, 1])

new_image = transforms(image)
new_image = new_image.numpy().transpose([1, 2, 0])
image = image.numpy().transpose([1, 2, 0])

plt.imshow(image)
plt.show()

plt.imshow(new_image)
plt.show()

print(image.shape, new_image.shape)