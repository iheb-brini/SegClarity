import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def split_components(attr,zero_threshold=0.05):
    """
    Split attribution into positive,negative and zero components
    """
    # this is a fix for old torch versions
    zero = torch.tensor(0., device=attr.device, dtype=attr.dtype)
    high = torch.tensor(0.9, device=attr.device, dtype=attr.dtype)
    
    pv= torch.where(attr > zero_threshold, attr, zero)
    nv= torch.where(attr < zero_threshold, -attr, zero)
    zv = torch.where(((attr > -zero_threshold) & (attr<zero_threshold)), high, zero)
    return pv,nv,zv

def generate_custom_heatmap_with_pil(attr,sign='all'):
    if sign == 'all':
        if attr.min() == 0:
            cmap = plt.colormaps['YlGn']
        else:
            cmap = plt.colormaps['RdYlGn']
            attr =  (attr + 1) / 2
        # Normalize the attribute data to the range [0, 1]
    
    elif sign=='pos':
        cmap = plt.colormaps["Greens"]
    elif sign=='neg':
        cmap = plt.colormaps['Reds']
    else:
        cmap = plt.colormaps['binary']        

    # Apply the colormap
    colored_data = cmap(attr)

    # Convert the data to a format suitable for PIL
    colored_data = (colored_data[:, :, :3] * 255).astype(np.uint8)  # Exclude the alpha channel
    heatmap_img = Image.fromarray(colored_data, 'RGB')
    return heatmap_img