import os
from scipy import stats as STS

from torch import bincount, diag, Tensor,ones_like,zeros_like
from numpy import nanmean

def isNaN(num):
    return num != num

def pcc(A, B):
    # Pearson correlation coefficient
    # '''
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    Asq, Bsq = A.view(1, -1).squeeze(0).cpu(), B.view(1, -1).squeeze(0).cpu()
    
    if Asq.isnan().any() or Bsq.isnan().any() or (Asq == 0).all() or (Bsq == 0).all():
        y = 0.
    else:
        y, _ = STS.pearsonr(Asq, Bsq)
        y = (y + 1) / 2

    return y


import torch

def pcc_cuda(A, B):
    """
    Compute the Pearson correlation coefficient between two PyTorch tensors A and B.

    Args:
    A (torch.Tensor): First input tensor.
    B (torch.Tensor): Second input tensor.

    Returns:
    float: The Pearson correlation coefficient, adjusted to be in the range [0, 1].
    """
    
    # Flatten the tensors.
    Asq, Bsq = A.view(-1), B.view(-1)
    
    # Check for NaN values in the tensors and null matrices.
    if torch.isnan(Asq).any() or torch.isnan(Bsq).any() or (Asq == 0).all() or (Bsq == 0).all():
        return 0.  # Return 0 if there are any NaNs.
    
    # Mean-center the tensors
    Asq_mean_centered = Asq - Asq.mean()
    Bsq_mean_centered = Bsq - Bsq.mean()
    
    # Compute covariance and the standard deviations
    covariance = (Asq_mean_centered * Bsq_mean_centered).mean()
    Asq_std = Asq.std()
    Bsq_std = Bsq.std()
    
    # Compute Pearson correlation coefficient
    pearson_correlation = covariance / (Asq_std * Bsq_std)
    
    # Normalize the coefficient to be in the range [0, 1]
    pearson_correlation_normalized = (pearson_correlation + 1) / 2

    return pearson_correlation_normalized.item()

def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist

def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """

    EPS = 1e-6
    A_inter_B = diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = jaccard.nanmean()  # the mean of jaccard without NaNs
    return avg_jacc, jaccard


def calculate_explanation_map(inp, saliency_map, normalize_map, device):

    if len(inp.shape) == 3:
        inp = inp.unsqueeze(0)
        
    if (saliency_map == 0).all():
        return zeros_like(inp).to(device)
    
    return inp.to(device)*normalize_map(saliency_map).to(device)


from typing import List
import numpy as np
class Square:
    """A square object with x, y coordinates and size,
    x: x coordinate of the square top left corner
    size: size of the square
    """
    def __init__(self,x,y, size):
        self.x = x
        self.y = y
        self.size = size
    
    def __repr__(self):
        return f"Square(x={self.x},y={self.y},size={self.size})"

def generate_random_squares(h,w,n_squares, size) -> List[Square]:
    squares = []
    for i in range(n_squares):
        x = np.random.randint(0,w)
        y = np.random.randint(0,h)
        squares.append(Square(x,y,size))
    return squares
