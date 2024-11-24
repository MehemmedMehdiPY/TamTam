# import warnings
# warnings.filterwarnings('ignore')

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# from torch.nn import functional as f
# import albumentations


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# class ProcessingSupporter:
#     def __init__(self):
#         """Class to support pre-processing techniques"""
#         pass

#     def free_outliers(self, image, whis=1.5):
#         """Outlier Mitigation"""
#         q1, q3 = np.quantile(image, q=[0.25, 0.75], axis=[-2, -1])
#         iqr = q3 - q1

#         lower = q1 - whis * iqr
#         upper = q3 + whis * iqr

#         lower = lower[:, np.newaxis, np.newaxis]
#         upper = upper[:, np.newaxis, np.newaxis]

#         image = np.clip(image, a_min=lower, a_max=upper)

#         return image

#     def min_max_scaler(self, image, eps=1e-16):
#         """Scaling"""
#         image_min = 0
#         image_max = 255
#         image = (image - image_min) / (image_max - image_min + eps)
#         return image

#     def split(self, indexes, train_size=0.75, seed=None):
#         """Dataset splitting"""
#         if seed:
#             torch.random.manual_seed(seed=seed)

#         train_indexes, val_indexes = random_split(indexes, lengths=[train_size, 1 - train_size])
#         return train_indexes, val_indexes

class CardImageDataset(Dataset):
    def __init__(self, root: str, mode:str = "train", transforms = None) -> None:
        super().__init__()

        self.root = root
        self.mode = mode
        self.transforms = transforms

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
        image = torch.tensor(image).to(torch.float32)
        
        return image, label

    def __len__(self):
        return len(self.images)



    # def __init__(self, root: str, colors: np.darray, to_loader: bool = False, transform: albumentations.core.composition.Compose = None, 
    #              transform_image: albumentations.core.composition.Compose = None, num_classes: int = 2, 
    #              train_size: float = 0.75, seed: int = None):
    #     """The class to support data preparation
    #     root:                       Dataset path
    #     colors:                     Unique color codes for classes
    #     to_loader:                  If True, some processing techniques will be added for data loader.
    #     transform:                  Used to support augmentation in both images and masks
    #     transform_image:            Used to support augmentation in only images
    #     num_classes:                The number of classes in the mask
    #     train_size:                 The measure to indicate the split size of training data
    #     seed:                       Randomness of dataset splitting           
    #     """
    #     super().__init__()

    #     self.root = root
    #     self.colors = colors
    #     self.to_loader = to_loader

    #     func = lambda image, mask: {'image': image, 'mask': mask}
    #     self.transform = (func if transform is None
    #                       else transform)

    #     func = lambda image: {'image': image}
    #     self.transform_image = (func if transform_image is None
    #                       else transform_image)

    #     self.num_classes = num_classes
    #     self.train_size = train_size
    #     self.seed = seed

    #     self.image_path = os.path.join(self.root, 'image')
    #     self.mask_path = os.path.join(self.root, 'mask')

    #     self.image_filenames = np.array(os.listdir(self.image_path))
    #     self.mask_filenames = np.array(os.listdir(self.mask_path))

    #     self.image_filenames.sort()
    #     self.mask_filenames.sort()

    #     self.mode = None

    #     if self.to_loader:
    #         indexes = list(range(len(self.image_filenames)))
    #         train_indexes, val_indexes = self.split(indexes=indexes, train_size=self.train_size, seed=self.seed)

    #         self.train_images = self.image_filenames[train_indexes]
    #         self.val_images = self.image_filenames[val_indexes]
    #         self.train_masks = self.mask_filenames[train_indexes]
    #         self.val_masks = self.mask_filenames[val_indexes]

    #         self.set_mode(mode='train')

    # def __getitem__(self, idx):
    #     image_filename = self.image_filenames[idx]
    #     mask_filename = self.mask_filenames[idx]

    #     image_filename = os.path.join(self.image_path, image_filename)
    #     mask_filename = os.path.join(self.mask_path, mask_filename)

    #     image = np.load(image_filename)
    #     mask = np.load(mask_filename)
        
    #     if self.to_loader:
    #         mask = mask[:, :, np.newaxis]

    #         out = self.transform(image=image, mask=mask)
    #         image, mask = out['image'], out['mask']

    #         out = self.transform_image(image=image)
    #         image = out['image']

    #         image = torch.tensor(image)
    #         mask = torch.tensor(mask).long()
    #         mask = f.one_hot(mask[:, :, 0], num_classes=self.num_classes)

    #         image = image.permute([2, 0, 1])
    #         mask = mask.permute([2, 0, 1])

    #         image = self.free_outliers(image)
    #         image = self.min_max_scaler(image)

    #         image = image.to(torch.float32)
    #         mask = mask.to(torch.float32)

    #     return image, mask

    # def __len__(self):
    #     return len(self.image_filenames)

    # def set_mode(self, mode):
    #     mode = mode.lower()

    #     if not self.to_loader:
    #         print('Warning! No mode can be set while to_loader=False')

    #     elif mode == 'train':
    #         self.image_filenames = self.train_images
    #         self.mask_filenames = self.train_masks
    #         self.mode = mode

    #     elif mode == 'val':
    #         self.image_filenames = self.val_images
    #         self.mask_filenames = self.val_masks
    #         self.mode = mode

    #     else:
    #         print('Warning! No mode can be set as {}'.format(mode))

    # def plot_image(self, idx):

    #     if self.to_loader:
    #         raise Exception('You can\'t use plot_image or plot_mask while to_loader is True')

    #     image, _ = self.__getitem__(idx)

    #     plt.imshow(image)
    #     plt.axis('off')
    #     plt.show()

    # def plot_mask(self, idx):

    #     if self.to_loader:
    #         raise Exception('You can\'t use plot_image or plot_mask while to_loader is True')

    #     _, mask = self.__getitem__(idx)
    #     mask = self.colors[mask]
        
    #     plt.imshow(mask)
    #     plt.axis('off')
    #     plt.show()
