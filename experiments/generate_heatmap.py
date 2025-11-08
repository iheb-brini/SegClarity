# %%
import os
from sys import path

path.append("..")
import pathlib

path.append(str(pathlib.Path(__file__).parent.parent))
from Modules.Dataset import generate_dataset

# %%
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from Modules.Dataset import generate_dataset
from Modules.Architecture import generate_model
from Modules.Visualization import (
    visualize_mask,
    visualize_image_and_mask,
)
from Modules.ModelXAI import generate_XAI_model

from Modules.Ablation.core import RISE
from Modules.Visualization.core import generate_heatmap
from Modules.Utils import Normalizations
import matplotlib.pyplot as plt
from Modules.Utils import clip_fixed_percentage

# %%
dataset_type = "UTP"
dataset_type = "splitAB1"

base_dir: str = f"datasets/{dataset_type}"
device = "cuda:0"
device = "cpu"

model_type = "unet"
MODEL_TAGS = ["from_scratch_models", "finetuned_models_minloss"]
model_tag = MODEL_TAGS[0] if dataset_type == "UTP" else MODEL_TAGS[1]

OUT_CHANNELS = 7 if dataset_type == "UTP" else 4

# Dataloader
if dataset_type == "splitAB1":
    _, _, test_loader = generate_dataset(
        dataset_type=dataset_type,
        base_dir=base_dir,
        **{
            "train_folder": "training-40",
            "validation_folder": "validation",
            "test_folder": "test",
        },
    )
else:
    _, _, test_loader = generate_dataset(
        dataset_type=dataset_type,
        base_dir=base_dir,
        **{
            "train_folder": "Training",
            "validation_folder": "Validation",
            "test_folder": "Test",
        },
    )


UTP_EPOCHS_DICT = {
    "lunet": {MODEL_TAGS[0]: 84, MODEL_TAGS[1]: 61},
    "unet": {MODEL_TAGS[0]: 85, MODEL_TAGS[1]: 43},
}


CB55_EPOCHS_DICT = {
    "lunet": {MODEL_TAGS[0]: 108, MODEL_TAGS[1]: 40},
    "unet": {MODEL_TAGS[0]: 60, MODEL_TAGS[1]: 87},
}


if dataset_type == "UTP":
    models_path = f"models/{dataset_type}/{model_type}/{model_tag}"
    model_name = f"model{UTP_EPOCHS_DICT[model_type][model_tag]}.pt"
else:
    models_path = f"models/{dataset_type}/{model_type}/{model_tag}"
    model_name = f"model{CB55_EPOCHS_DICT[model_type][model_tag]}.pt"

print(f"Loading model from {models_path}")
print(f"Model name: {model_name}")


# %%
print("Loading model...")
model = (
    generate_model(
        model_type=model_type,
        out_channels=OUT_CHANNELS,
        load_from_checkpoint=True,
        models_path=models_path,
        checkpoint_name=model_name,
    )
    .eval()
    .to(device)
)

# Prepare model for XAI
model = generate_XAI_model(model=model, device=device)


from Modules.Attribution.core import generateAttributionsOnInput
from Modules.Utils import clip_fixed_percentage
from Modules.Visualization.core import generate_heatmap
from Modules.Utils import Normalizations
import matplotlib.pyplot as plt


debug = False
kwargs = {"layer": model.final_layer}
methods = ["GuidedGradCam", "LRP", "InputXGradient", "DeepLift"]
methods = ["DeepLift"]

for idx, (image_batch, mask_batch, paths) in tqdm(enumerate(test_loader)):
    image_batch = image_batch.to(device, dtype=torch.float)
    mask_batch = mask_batch.to(device, dtype=torch.long)
    image_name = paths[0][0].split("/")[-1].split(".")[0]
    if "53v" not in image_name and "23v" not in image_name:
        continue
    available_target = mask_batch.unique()
    for method in tqdm(methods):
        for selected_target in tqdm(available_target):
            if selected_target == 0:
                continue
            attribution_path = f"attributions/{dataset_type}/{model_type}/{model_tag}"
            file_name = f"{image_name}_{method}_{selected_target}.pt"
            if os.path.exists(f"{attribution_path}/{file_name}"):
                continue
            attr = (
                generateAttributionsOnInput(
                    image_batch,
                    model,
                    selected_target,
                    method,
                    device=device,
                    debug=debug,
                    **kwargs,
                )
                .detach()
                .cpu()
            )
            attr_sum = attr.sum(dim=1, keepdim=True).squeeze()
            os.makedirs(attribution_path, exist_ok=True)
            torch.save(attr_sum, f"{attribution_path}/{file_name}")
