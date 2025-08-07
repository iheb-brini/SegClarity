import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from torchvision.datasets.folder import pil_loader
from .tools import (
    get_sampler,
    rgb_to_2D_label,
    generate_paths,
    generate_dirs,
    dataset_status,
)
from .constants import (
    IMG_TAG,
    MASK_TAG,
    UTP_BASE_DIR,
    SPLITAB1_BASE_DIR,
    UTP_LABELS,
    SPLITAB1_LABELS,
)


class DIVA(Dataset):

    def __init__(self, imagePaths, maskPaths, labels):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.labels = labels

    def __getitem__(self, idx):

        # read data
        img_path = self.imagePaths[idx]
        mask_path = self.maskPaths[idx]
        read_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = 2 * ((read_img - read_img.min()) / (read_img.max() - read_img.min())) - 1
        mask = pil_loader(mask_path)
        # mask to numpy
        np_mask_diva = np.array(mask)
        # mask from RGB to labels (HxW)
        mask = rgb_to_2D_label(np_mask_diva, self.labels)
        # Image and mask normalization
        Transforms = transforms.Compose([transforms.ToTensor()])
        img = Transforms(img)
        mask = torch.from_numpy(mask).long()

        return img, mask, (img_path, mask_path)

    def __len__(self):
        return len(self.imagePaths)


def create_dataloaders(
    base_dir=UTP_BASE_DIR,
    img_tag=IMG_TAG,
    mask_tag=MASK_TAG,
    labels=UTP_LABELS,
    **kwargs
):
    dirs = generate_dirs(base_dir, img_tag, mask_tag, **kwargs)
    dataset_status(dirs[0], dirs[2], dirs[4])

    (
        train_img_paths,
        train_mask_paths,
        val_img_paths,
        val_mask_paths,
        test_img_paths,
        test_mask_paths,
    ) = generate_paths(*dirs)

    train_dataset = DIVA(train_img_paths, train_mask_paths, labels)
    val_dataset = DIVA(val_img_paths, val_mask_paths, labels)
    test_dataset = DIVA(test_img_paths, test_mask_paths, labels)
    train_loader = DataLoader(
        train_dataset, batch_size=1, sampler=get_sampler(train_dataset), num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset)

    return train_loader, val_loader, test_loader


def generate_splitAB1_dataset(base_dir=SPLITAB1_BASE_DIR, **kwargs):
    return create_dataloaders(
        base_dir=base_dir,
        img_tag="codex",
        mask_tag="truthL",
        labels=SPLITAB1_LABELS,
        **kwargs
    )


def generate_UTP_dataset(base_dir=UTP_BASE_DIR, **kwargs):
    return create_dataloaders(
        base_dir=base_dir,
        img_tag="images",
        mask_tag="masks",
        labels=UTP_LABELS,
        **kwargs
    )


dataset_creator = {
    "splitAB1": generate_splitAB1_dataset,
    "UTP": generate_UTP_dataset,
}

dataset_creator = {
    "splitAB1": {
        "dataset": generate_splitAB1_dataset,
        "train_folder": "training-10",
        "validation_folder": "validation",
        "test_folder": "test",
    },
    "UTP": {
        "dataset": generate_UTP_dataset,
        "train_folder": "Training",
        "validation_folder": "Validation",
        "test_folder": "Test",
    },
}

base_dir: str = "/HOME/brinii/Projects/IJDAR2024/sandbox/datasets/UTP"


def generate_dataset(dataset_type, base_dir, **kwargs):
    config = dataset_creator[dataset_type]

    train_folder = kwargs.get("train_folder", config["train_folder"])
    validation_folder = kwargs.get("validation_folder", config["validation_folder"])
    test_folder = kwargs.get("test_folder", config["test_folder"])

    return config["dataset"](
        base_dir=base_dir,
        train_folder=train_folder,
        validation_folder=validation_folder,
        test_folder=test_folder,
    )
