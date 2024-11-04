import torch
import torchvision.transforms.functional as TF
from typing import Dict, Tuple, Union

class BuildingSegmentationTransform:
    """Base class for building segmentation transforms."""
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the transform to both image and mask.
        
        Args:
            image: Input image tensor
            mask: Input mask tensor
            
        Returns:
            Transformed image and mask tensors
        """
        raise NotImplementedError

class Compose:
    """Compose multiple transforms together."""
    def __init__(self, transforms: list):
        self.transforms = transforms
        
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask

class Normalize(BuildingSegmentationTransform):
    """Normalize the image using mean and std."""
    def __init__(self, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image = TF.normalize(image, self.mean, self.std)
        return image, mask