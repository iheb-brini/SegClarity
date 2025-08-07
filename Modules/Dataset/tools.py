import torch
from torch.utils.data import RandomSampler
import numpy as np
import glob
import os
from .constants import SPLITAB1_LABELS,UTP_LABELS,IMG_TAG,MASK_TAG

def generate_dirs(base_dir,img_tag=IMG_TAG, mask_tag=MASK_TAG,**kwargs):
    train_folder = kwargs.get("train_folder","train")
    validation_folder = kwargs.get("validation_folder","validation")
    test_folder = kwargs.get("test_folder","test")
    img_dir = f"{base_dir}/{img_tag}"
    mask_dir = f"{base_dir}/{mask_tag}"

    x_train_dir = os.path.join(img_dir, train_folder)
    y_train_dir = os.path.join(mask_dir, train_folder)
    x_val_dir = os.path.join(img_dir, validation_folder)
    y_val_dir = os.path.join(mask_dir, validation_folder)
    x_test_dir = os.path.join(img_dir, test_folder)
    y_test_dir = os.path.join(mask_dir, test_folder)

    return (x_train_dir,
            y_train_dir,
            x_val_dir,
            y_val_dir,
            x_test_dir,
            y_test_dir)


def dataset_status(x_train_dir, x_valid_dir, x_test_dir):
    print('the number of image/label in the train: ',
          len(os.listdir(x_train_dir)))
    print('the number of image/label in the validation: ',
          len(os.listdir(x_valid_dir)))
    print('the number of image/label in the test: ', len(os.listdir(x_test_dir)))


def generate_paths(x_train_dir, y_train_dir,
                   x_val_dir, y_val_dir,
                   x_test_dir, y_test_dir):
    train_img_paths = glob.glob(os.path.join(x_train_dir, "*.jpg"))
    train_mask_paths = glob.glob(os.path.join(y_train_dir, "*.gif"))
    val_img_paths = glob.glob(os.path.join(x_val_dir, "*.jpg"))
    val_mask_paths = glob.glob(os.path.join(y_val_dir, "*.gif"))
    test_img_paths = glob.glob(os.path.join(x_test_dir, "*.jpg"))
    test_mask_paths = glob.glob(os.path.join(y_test_dir, "*.gif"))

    train_img_paths.sort()
    train_mask_paths.sort()

    val_img_paths.sort()
    val_mask_paths.sort()

    test_img_paths.sort()
    test_mask_paths.sort()
    
    return (train_img_paths, train_mask_paths,
            val_img_paths, val_mask_paths,
            test_img_paths, test_mask_paths)
    
def get_sampler(dataset, seed=123):
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(dataset, generator=generator)
    return sampler


# mask from RGB to labels (HxW)
def rgb_to_2D_label(mask,labels):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(mask.shape, dtype=np.uint8)
    for idx,label in enumerate(labels):
        label_seg[np.all(mask == label, axis=-1)] = idx
    # Just take the first channel, no need for all 3 channels
    label_seg = label_seg[:, :, 0]

    return label_seg
