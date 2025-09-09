import torch
import numpy as np
import matplotlib.pyplot as plt
from .constants import void_classes, valid_classes, class_map, n_classes, label_colours, ignore_index
import albumentations as A
from albumentations.pytorch import ToTensorV2


def encode_segmap(mask):
    """
    Remove unwanted classes and rectify the labels of wanted classes
    """
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask


def decode_segmap(temp):
    """
    Convert grayscale segmentation map to color
    """
    if torch.is_tensor(temp):
        temp = temp.numpy()
    
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def get_transforms(image_size=(256, 512), augment=True):
    """
    Get data transforms for training/validation
    """
    if augment:
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return transform


def visualize_batch(images, masks, num_samples=4, figsize=(16, 8)):
    """
    Visualize a batch of images and their corresponding masks
    
    Args:
        images: Batch of images (B, C, H, W)
        masks: Batch of masks (B, H, W)
        num_samples: Number of samples to visualize
        figsize: Figure size
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(2, num_samples, figsize=figsize)
    
    for i in range(num_samples):
        # Denormalize image
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Get mask
        mask = masks[i].numpy()
        
        # Decode mask to color
        mask_color = decode_segmap(mask)
        
        # Plot
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Image {i}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask_color)
        axes[1, i].set_title(f'Mask {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def get_dataset_info():
    """
    Print information about the Cityscapes dataset
    """
    from .constants import class_names, class_map, n_classes, ignore_index
    
    print(f"Number of classes: {n_classes}")
    print(f"Ignore index: {ignore_index}")
    print("\nClass names:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    print(f"\nClass mapping:")
    for original_id, new_id in class_map.items():
        print(f"  {original_id} -> {new_id}")
