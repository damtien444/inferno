import torch
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte


def concatenate_image_batch_to_tall_image(images : torch.Tensor) -> torch.Tensor:
    return images.reshape([-1,] + list(images.shape[2:]))


def concatenate_image_batch_to_wide_image(images : torch.Tensor) -> torch.Tensor:
    images = images.split(1, dim=0)
    images = [torch.squeeze(i) for i in images]
    images = torch.cat(images, 1)
    return images


def torchFloatToNpUintImage(image: torch.Tensor) -> np.ndarray:
    return img_as_ubyte(rescale_intensity(image.cpu().numpy() * 255, in_range='uint8'))


def numpy_image_to_torch(img : np.ndarray) -> torch.Tensor:
    img = img.transpose([2, 0, 1])
    return torch.from_numpy(img)