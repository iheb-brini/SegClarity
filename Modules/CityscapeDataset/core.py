import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from typing import Any, Tuple
from PIL import Image
import numpy as np
from .tools import encode_segmap, get_transforms
from .constants import n_classes


class CityscapesDataset(Cityscapes):
    """
    Custom Cityscapes dataset with proper preprocessing
    """
    def __init__(self, root, split='train', mode='fine', target_type='semantic', 
                 transforms=None, image_size=(256, 512)):
        super().__init__(root, split, mode, target_type, transforms)
        self.transforms = transforms
        self.image_size = image_size
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        name = self.images[index]
        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]
        
        if self.transforms is not None:
            transformed = self.transforms(image=np.array(image), mask=np.array(target))
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Convert to tensors if no transforms
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(np.array(target)).long()
        
        # Encode the segmentation map
        mask = encode_segmap(mask)
        
        return image, mask, name


def create_cityscapes_dataloaders(root='cityscapes', 
                                 batch_size=8, 
                                 num_workers=4, 
                                 image_size=(256, 512),
                                 augment_train=True):
    """
    Create train, validation, and test dataloaders for Cityscapes
    
    Args:
        root: Path to Cityscapes dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloading
        image_size: Tuple of (height, width) for resizing
        augment_train: Whether to apply augmentation to training data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create transforms
    train_transform = get_transforms(image_size, augment=augment_train)
    val_transform = get_transforms(image_size, augment=False)
    test_transform = get_transforms(image_size, augment=False)
    
    # Create datasets
    train_dataset = CityscapesDataset(
        root=root, 
        split='train', 
        mode='fine', 
        target_type='semantic',
        transforms=train_transform,
        image_size=image_size
    )
    
    val_dataset = CityscapesDataset(
        root=root, 
        split='val', 
        mode='fine', 
        target_type='semantic',
        transforms=val_transform,
        image_size=image_size
    )
    
    test_dataset = CityscapesDataset(
        root=root, 
        split='test', 
        mode='fine', 
        target_type='semantic',
        transforms=test_transform,
        image_size=image_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
