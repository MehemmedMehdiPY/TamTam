import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class ProcessingSupporter:
    def __init__(self):
        """Class to support pre-processing techniques"""
        pass

    def free_outliers(self, image, whis=1.5):
        """Outlier Mitigation"""
        q1, q3 = np.quantile(image, q=[0.25, 0.75], axis=[-2, -1])
        iqr = q3 - q1

        lower = q1 - whis * iqr
        upper = q3 + whis * iqr

        lower = lower[:, np.newaxis, np.newaxis]
        upper = upper[:, np.newaxis, np.newaxis]

        image = np.clip(image, a_min=lower, a_max=upper)
        return image

    def min_max_scaling(self, image, eps=1e-16):
        """Scaling"""
        image_min = 0
        image_max = 255
        image = (image - image_min) / (image_max - image_min + eps)
        return image
    
class CardImageDataset(Dataset, ProcessingSupporter):
    def __init__(self, root: str, mode:str = "train", transforms = None) -> None:
        super().__init__()

        self.root = root
        self.mode = mode
        self.transforms = (
            transforms if transforms is not None
            else lambda x: x)

        self.folder = os.path.join(self.root, self.mode)

        self.classes = os.listdir(self.folder)
        self.classes.sort()

        self.images = []
        self.labels = []

        for cls_id in range(len(self.classes)):
            cls_images = os.path.join(self.folder, self.classes[cls_id])
            for image in os.listdir(cls_images):
                image = os.path.join(cls_images, image)
                self.images.append(image)

                label = np.zeros(len(self.classes))
                label[cls_id] = 1

                self.labels.append(label.tolist())
        
        self.labels = torch.tensor(self.labels).to(torch.float32)
        
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = np.asarray(Image.open(image))
        image = image.astype(np.float32)
        image = image.transpose([2, 0, 1])
        image = self.free_outliers(image=image)    
        image = torch.tensor(image).to(torch.float32)
        image = self.transforms(image)
        image = self.min_max_scaling(image)
        return image, label

    def __len__(self):
        return len(self.images)

