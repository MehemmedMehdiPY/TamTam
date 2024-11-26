import torch
from torchvision.transforms import v2

class RandomResizedCrop():
    def __init__(self, p, size, scale, ratio):
        """The existing RandomResizedCrop didn't have p parameter. It is because scale parameter already simulates similar behaviors.
        However, on its own, obtaining the original image without random cropping was challenging. Therefore,
        I decided to add p parameter to reduce the frequency of transformation."""
        self.p = p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.transform = v2.RandomResizedCrop(size=self.size, scale=self.scale, ratio=self.ratio)

    def __call__(self, image):
        p = (torch.rand(1) * 10).ceil() / 10
        image = (image if p >= self.p
                 else self.transform(image)
                 )
        return image
