from torchvision import models
from torch.cuda import is_available
from .lunet import lunet_model
from .unet import unet_model

from .constants import (
    DEFAULT_LUNET_MODEL_PATH,
    DEFAULT_UNET_MODEL_PATH,
    DEFAULT_DEVICE,
)
import torch


def generate_fcn_model(device=DEFAULT_DEVICE):
    return models.segmentation.fcn_resnet101(pretrained=True).to(device).eval()


def generate_unet_model(
    out_channels,
    checkpoint_name,
    models_path=DEFAULT_UNET_MODEL_PATH,
    device=DEFAULT_DEVICE,
):
    model = unet_model(out_channels).to(device)
    try:
        checkpoint = torch.load(f"{models_path}/{checkpoint_name}", map_location=device)
        # load model weights state_dict
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    except:
        print("Checkpoint not found!")


def generate_lunet_model(
    out_channels,
    checkpoint_name,
    models_path=DEFAULT_LUNET_MODEL_PATH,
    device=DEFAULT_DEVICE,
):
    model = lunet_model(out_channels).to(device)
    try:
        checkpoint = torch.load(f"{models_path}/{checkpoint_name}", map_location=device)
        # load model weights state_dict
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    except:
        print("Checkpoint not found!")


# CHANGE THIS TO ADD A NEW MODEL
model_creator = {
    "unet": {"arch": unet_model, "model_path": DEFAULT_UNET_MODEL_PATH},
    "lunet": {"arch": lunet_model, "model_path": DEFAULT_LUNET_MODEL_PATH},
}


def generate_model(
    model_type,
    out_channels,
    device=DEFAULT_DEVICE,
    load_from_checkpoint=False,
    models_path=None,
    checkpoint_name=None,
):
    if model_type not in model_creator.keys():
        raise ValueError(
            f"Model type: {model_type} not recognized from {list(model_creator.keys())}"
        )
    model = model_creator[model_type]["arch"](out_channels=out_channels).to(device)
    if load_from_checkpoint:
        try:
            models_path = (
                model_creator[model_type]["model_path"]
                if models_path is None
                else models_path
            )

            print(f"{models_path}/{checkpoint_name}")
            checkpoint = torch.load(
                f"{models_path}/{checkpoint_name}", map_location=device
            )
            # load model weights state_dict
            model.load_state_dict(checkpoint["model_state_dict"])
        except:
            ValueError("Checkpoint not found!")
    return model
