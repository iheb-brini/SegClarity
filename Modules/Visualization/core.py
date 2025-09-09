import torch
from torchvision.transforms import ToPILImage
from Modules.Utils import Normalizations
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import numpy as np
from torch.nn.functional import interpolate

from .tools import split_components,generate_custom_heatmap_with_pil

def generate_heatmap(
    attr,
    ax=None,
    fig=None,
    title=None,
    save_file=None,
    normalization_name=Normalizations.normalize_scaler,
    normalizer_args=[],
    mode="plt",
    **kwargs
):

    add_bar = kwargs.get("add_bar", False)
    if type(attr) == type(torch.zeros(1)):
        V = attr.clone().squeeze().detach().cpu()
    else:
        V = attr.clone()

    if normalization_name is not None:
        normalizer = Normalizations.pick(normalization_name)
        if V.sum().item() != 0:
            V = normalizer(V.squeeze().detach().cpu(), *normalizer_args)
        else:
            V = V.numpy()

    if ax is None:
        ax = plt.gca()
    if V.max() == 0:
        if not normalization_name == Normalizations.identity:
            V[-1,-1] = -1
        img = ax.imshow(-V, cmap="YlOrRd")
    elif V.min() == 0:
        if not normalization_name == Normalizations.identity:
            V[-1,0] = 1
        img = ax.imshow(V, cmap="YlGn")
    else:
        if not normalization_name == Normalizations.identity:
            V[-1,0] = 1
            V[-1,-1] = -1
        img = ax.imshow(V, cmap="RdYlGn")

    ax.axis("off")
    if add_bar:
        fig.colorbar(img)
    if title is not None:
        fontsize = kwargs.get("fontsize", None)
        if fontsize is not None:
            ax.set_title(title, fontsize=fontsize)
        else:
            ax.set_title(title)
    if save_file is not None:
        plt.gcf().savefig(save_file, bbox_inches="tight", pad_inches=0.0, dpi=200)
    return ax


def generate_heatmap_viz(attr,img_path,
                         nr=Normalizations.normalize_non_shifting_zero,
                         figtite=None,
                         savefig=None,
                         **kwargs):
    no_title = kwargs.get('no_title',False)
    normalizer = Normalizations.pick(nr)
    norm_attr = normalizer(attr)
    titles = kwargs.get('titles',['all','pos','neg','blended'])
    title_loc = kwargs.get('title_loc',0.95)
    no_blend = kwargs.get('no_blend',False)
    pv,nv,_ = split_components(norm_attr)

    pos_image = generate_custom_heatmap_with_pil(pv,'pos')
    neg_image = generate_custom_heatmap_with_pil(nv,'neg')
    all_image = generate_custom_heatmap_with_pil(norm_attr,'all')

    if no_blend:
        fig,axs = plt.subplots(1,3)
    else:
        fig,axs = plt.subplots(1,4)
    fig.set_size_inches(18,6)

    if figtite is not None:
        fig.suptitle(figtite,y=title_loc)
    plots = [all_image,pos_image,neg_image]

    for i,ax in enumerate(axs):
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if i == 3:
            img = Image.open(img_path)
            size_org = img.size
            ax.imshow(img)
            ax.imshow(interpolate(norm_attr.unsqueeze(0).unsqueeze(1),(size_org[1],size_org[0])).squeeze(), cmap='RdYlGn',alpha=0.3)
        else:
            ax.imshow(plots[i])
        
        if not no_title:
            ax.set_title(titles[i])


    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight', pad_inches = 0.0,dpi=200)

    return fig,axs

def generate_heatmap_components(attr,figtite,savefig,**kwargs):
    zero_threshold = kwargs.get('zero_threshold',None)
    nr = Normalizations.normalize_non_shifting_zero
    normalizer = Normalizations.pick(nr)
    norm_attr = normalizer(attr)

    if zero_threshold is not None:
        pv,nv,zv = split_components(norm_attr,zero_threshold=zero_threshold)
    else:
        pv,nv,zv = split_components(norm_attr)

    pos_image = generate_custom_heatmap_with_pil(pv,'pos')
    neg_image = generate_custom_heatmap_with_pil(nv,'neg')
    zero_image = generate_custom_heatmap_with_pil(zv,'zero')
    all_image = generate_custom_heatmap_with_pil(norm_attr,'all')

    #dpi=100
    #h,w = norm_attr.shape
    #size = [float(i)/dpi for i in (h,w)]

    fig,axs = plt.subplots(1,4)
    fig.set_size_inches(18,6)


    fig.suptitle(figtite,y=0.95)
    plots = [all_image,pos_image,neg_image,zero_image]
    titles = ['all','pos','neg','zero']

    for i,ax in enumerate(axs):
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.imshow(plots[i])
        ax.set_title(titles[i])
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight', pad_inches = 0.1,dpi=200)


def visualize_mask(mask_batch,dataset_type='UTP',fig=None,ax=None,show=True,save_file=None):
    """
    Visualizes a single-channel mask with classes represented by unique values.

    Parameters:
    - mask_batch: a torch.Tensor of shape [1, height, width] with unique values for each class

    This function will display the mask with a unique color for each class.
    """
    if mask_batch.dim() == 3 and mask_batch.size(0) == 1:
        # Squeeze the mask_batch if it has a singleton dimension at the start
        mask = mask_batch.squeeze(0)
    else:
        raise ValueError("mask_batch should be of shape [1, height, width]")

    # Define a color map for visualizing different classes
    # Adjust these colors to your preference or add more for more classes
    if dataset_type == 'UTP':
        cmap = np.array(
            [
                [0, 0, 0],  # BLACK
                [255, 0, 0],  # RED
                [0, 255, 0],  # GREEN
                [0, 0, 255],  # BLUE
                [255, 255, 0], # YELLOW
                [255, 0, 255], # MAGENTA
                [0, 255, 255], # CYAN
                [255,255,255]
            ]
        )  # Class 3: Blue
    else:        
        cmap = np.array(
            [
                [0, 0, 0],  # black
                [255, 255, 0],  # yellow
                [255, 0, 255], # magenta
                [0, 255, 255], # cyan,
                [255,255,255]
            ]
        )  # Class 3: Blue
    cmap = cmap / 255.0  # Normalize to [0,1] for matplotlib

    # Convert the mask to a numpy array for visualization
    mask_np = mask.numpy()

    # Create an RGB image where each class is mapped to a specific color
    rgb_image = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.float32)
    for i in range(len(cmap)):
        rgb_image[mask_np == i] = cmap[i]

    # Plot the image
    if fig is None and ax is None:
        fig,ax = plt.subplots()
    
    ax.imshow(rgb_image)
    ax.axis("off")  # Turn off axis numbers and ticks
    if show:
        fig.tight_layout()
        fig.show()

    if save_file is not None:
        plt.gcf().savefig(save_file, bbox_inches='tight', pad_inches = 0.0, dpi=200)

    return ax

def visualize_image(image_batch):
    fig,axes = plt.subplots()
    axes.imshow(image_batch.squeeze().permute([1,2,0]).detach().cpu().numpy());axes.axis("off");axes.set_title('Image')
    fig.tight_layout()
    fig.show()




def visualize_image_and_mask(image_batch,mask_batch,dataset_type='UTP',fig = None, axes = None,save_file=None):
    if mask_batch.dim() == 3 and mask_batch.size(0) == 1:
        # Squeeze the mask_batch if it has a singleton dimension at the start
        mask = mask_batch.squeeze(0)
    else:
        raise ValueError("mask_batch should be of shape [1, height, width]")

    if dataset_type == 'UTP':
        cmap = np.array(
            [
                [0, 0, 0],  # BLACK
                [255, 0, 0],  # RED
                [0, 255, 0],  # GREEN
                [0, 0, 255],  # BLUE
                [255, 255, 0], # YELLOW
                [255, 0, 255], # MAGENTA
                [0, 255, 255], # CYAN
                [255,255,255]
            ]
        )  # Class 3: Blue
    else:        
        cmap = np.array(
            [
                [0, 0, 0],  # black
                [255, 255, 0],  # yellow
                [255, 0, 255], # magenta
                [0, 255, 255], # cyan,
                [255,255,255]
            ]
        )  # Class 3: Blue
    cmap = cmap / 255.0  # Normalize to [0,1] for matplotlib

    # Convert the mask to a numpy array for visualization
    mask_np = mask.numpy()

    # Create an RGB image where each class is mapped to a specific color
    rgb_image = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.float32)
    for i in range(len(cmap)):
        rgb_image[mask_np == i] = cmap[i]

    # Plot the image
    if fig is None and axes is None:
        fig,axes = plt.subplots(1,2)
    # Plot the image

    # Restore original image
    image_batch  = ((image_batch.squeeze().detach().cpu() + 1) / 2)
    image_batch  = image_batch * (image_batch.max() - image_batch.min()) + image_batch.min()

    axes[0].imshow(image_batch.squeeze().permute([1,2,0]).detach().cpu().numpy());axes[0].axis("off");axes[0].set_title('Image')
    axes[1].imshow(rgb_image);axes[1].axis("off");axes[1].set_title('Mask')
    
    fig.tight_layout()
    fig.show()

    if save_file is not None:
        plt.gcf().savefig(save_file, bbox_inches='tight', pad_inches = 0.0, dpi=200)

