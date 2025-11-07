from Modules.Evaluation.core import DiceMetric, Metric
from Modules.Evaluation.tools import _fast_hist, jaccard_index

import torch
import torch.nn.functional as F
from typing import Literal, Optional, Dict, Any, List
from torch import Tensor
from Modules.Ablation.tools import generate_mask_from_grid
from typing import Tuple
from tqdm import tqdm
import logging


def evaluated_single_masked_prediction(
    pred: torch.Tensor,
    pred_on_mask: torch.Tensor,
    target_class: int,
    score_mode: Literal["dice", "iou"] = "dice",
    num_classes: int = 4,
) -> float:
    """Evaluate the masked prediction using the specified score mode.

    Args:
        pred (torch.Tensor): The predicted mask.
        pred_on_mask (torch.Tensor): The predicted mask on the masked image.
        target_class (int): The target class.
        score_mode (str, optional): The score mode. Defaults to "dice".
    """
    if score_mode == "dice":
        dice_metric = DiceMetric()
        dice_gt_pred = dice_metric.calculate(pred, pred_on_mask, target_class)
        return dice_gt_pred
    elif score_mode == "iou":
        hist = _fast_hist(
            pred.squeeze().detach().cpu(),
            pred_on_mask.squeeze().detach().cpu(),
            num_classes=num_classes,
        )
        _, jacc = jaccard_index(hist)
        return jacc[target_class].item()
    else:
        raise ValueError(f"Invalid score mode: {score_mode}")


def generate_output_prediction(model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Generate the output prediction for the given model and input.

    Args:
        model (torch.nn.Module): The model to generate the output prediction.
        X (torch.Tensor): The input to the model.

    Returns:
        torch.Tensor: The output prediction.
    """
    out = model(X)
    out_max = torch.argmax(out, dim=1, keepdim=True)
    return out_max


def generate_masked_image(X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Generate the masked input tensor.

    Args:
        X (torch.Tensor): The input to the model.
        mask (torch.Tensor): The mask to apply to the input.

    Returns:
        torch.Tensor: The masked input tensor.
    """
    masked_X = mask.to(X.device) * X
    return masked_X


class RISE:
    """RISE implementation for saliency map generation."""

    def __init__(
        self, score_mode: Literal["dice", "iou"] = "dice", num_classes: int = 4
    ):
        """Initialize RISE with score mode and number of classes."""
        self.score_mode = score_mode
        self.num_classes = num_classes

    def generate_masks(
        self,
        num_masks: int,
        image_size: Tuple[int, int],
        grid_size: Tuple[int, int],
        prob_thresh: float,
    ):
        """Generate masks for RISE saliency map generation."""
        self.masks = torch.zeros(num_masks, 1, image_size[0], image_size[1])
        for i in range(num_masks):
            self.masks[i] = generate_mask_from_grid(image_size, grid_size, prob_thresh)
        return self.masks

    def compute_saliency(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        target_class: int,
        batch_size: int = 1,
    ):
        """Compute the saliency map for the given model and input.

        Args:
            model (torch.nn.Module): The model to compute the saliency map.
            X (torch.Tensor): The input to the model.
            target_class (int): The target class.
            batch_size (int): The batch size.

        Returns:
            tuple: The saliency map, the high scores and the low scores.
        """
        saliency = torch.zeros(1, X.shape[-2], X.shape[-1])
        out_max = generate_output_prediction(model, X)
        scores = []

        num_masks = self.masks.shape[0]
        num_batches = (num_masks + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_masks)
            batch_masks = self.masks[start_idx:end_idx]

            # Generate masked images for the batch
            batch_masked_X = torch.stack(
                [
                    generate_masked_image(X, batch_masks[i])
                    for i in range(batch_masks.shape[0])
                ]
            )

            # Process batch through model
            batch_masked_X = batch_masked_X.squeeze(1)
            out_on_mask_max = generate_output_prediction(model, batch_masked_X)

            # Evaluate each mask in the batch
            for i in range(batch_masks.shape[0]):
                score = evaluated_single_masked_prediction(
                    out_max,
                    out_on_mask_max[i : i + 1],
                    target_class=target_class,
                    score_mode=self.score_mode,
                    num_classes=self.num_classes,
                )
                scores.append(score)
                saliency += score * batch_masks[i]
        scores = torch.tensor(scores)  # [N]
        eps = 1e-8  # avoid division by zero
        saliency = saliency / (scores.sum() + eps)
        return saliency, scores


def dilate_mask(mask: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """Dilate a binary mask using max pooling.

    Args:
        mask (torch.Tensor): Binary mask to dilate.
        kernel_size (int): Size of the dilation kernel.

    Returns:
        torch.Tensor: Dilated mask.
    """
    # Use max pooling for dilation with proper padding
    padding = kernel_size // 2
    dilated = F.max_pool2d(
        mask.unsqueeze(0) if mask.dim() == 2 else mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    return dilated.squeeze(0) if mask.dim() == 2 else dilated


def total_variation_loss(mask: torch.Tensor) -> torch.Tensor:
    """Compute total variation loss for smoothness regularization.

    Args:
        mask (torch.Tensor): The mask tensor.

    Returns:
        torch.Tensor: Total variation loss.
    """
    # Compute gradients in x and y directions
    diff_x = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])
    diff_y = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :])

    tv_loss = diff_x.sum() + diff_y.sum()
    return tv_loss


def compute_weighted_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_class: int,
    alpha_fg: float = 2.0,
    alpha_bg: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute weighted Dice loss for foreground and background.

    Args:
        pred (torch.Tensor): Predicted segmentation.
        target (torch.Tensor): Target segmentation.
        target_class (int): The target class to evaluate.
        alpha_fg (float): Weight for foreground Dice.
        alpha_bg (float): Weight for background Dice.
        epsilon (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Weighted Dice loss.
    """
    # Convert to binary masks for target class
    pred_fg = (pred == target_class).float()
    target_fg = (target == target_class).float()

    pred_bg = (pred != target_class).float()
    target_bg = (target != target_class).float()

    # Compute Dice for foreground
    intersection_fg = (pred_fg * target_fg).sum()
    dice_fg = (2.0 * intersection_fg + epsilon) / (
        pred_fg.sum() + target_fg.sum() + epsilon
    )

    # Compute Dice for background
    intersection_bg = (pred_bg * target_bg).sum()
    dice_bg = (2.0 * intersection_bg + epsilon) / (
        pred_bg.sum() + target_bg.sum() + epsilon
    )

    # Weighted Dice score
    weighted_dice = (alpha_fg * dice_fg + alpha_bg * dice_bg) / (alpha_fg + alpha_bg)

    # Return loss (1 - dice)
    return 1.0 - weighted_dice


class MiSuRe:
    """MiSuRe (Minimally Sufficient Region) implementation for saliency map generation.

    MiSuRe is a two-stage model-agnostic explainability method for image segmentation:
    - Stage 1: Generate sufficient region by dilating the predicted object until successful segmentation
    - Stage 2: Optimize mask to find minimally sufficient region through gradient-based optimization

    Reference: https://arxiv.org/abs/2406.12173
    """

    def __init__(
        self,
        score_mode: Literal["dice", "iou"] = "dice",
        num_classes: int = 4,
        lambda_sparsity: float = 0.01,
        gamma_tv: float = 0.01,
        learning_rate: float = 0.1,
        alpha_fg: float = 2.0,
        alpha_bg: float = 1.0,
        tau: float = 0.9,
        mask_size: Tuple[int, int] = (224, 224),
        n_iterations: int = 100,
        dilation_kernel_size: int = 7,
    ):
        """Initialize MiSuRe with hyperparameters.

        Args:
            score_mode (str): Score mode for evaluation ('dice' or 'iou').
            num_classes (int): Number of segmentation classes.
            lambda_sparsity (float): Weight for sparsity loss.
            gamma_tv (float): Weight for total variation loss.
            learning_rate (float): Learning rate for optimization.
            alpha_fg (float): Weight for foreground Dice preservation.
            alpha_bg (float): Weight for background Dice preservation.
            tau (float): Dice threshold for sufficient region.
            mask_size (Tuple[int, int]): Size to resize masks to.
            n_iterations (int): Number of optimization iterations.
            dilation_kernel_size (int): Size of dilation kernel.
        """
        self.score_mode = score_mode
        self.num_classes = num_classes
        self.lambda_sparsity = lambda_sparsity
        self.gamma_tv = gamma_tv
        self.learning_rate = learning_rate
        self.alpha_fg = alpha_fg
        self.alpha_bg = alpha_bg
        self.tau = tau
        self.mask_size = mask_size
        self.n_iterations = n_iterations
        self.dilation_kernel_size = dilation_kernel_size

    def generate_sufficient_region(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        target_class: int,
        max_dilations: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Generate sufficient region (Stage 1) by dilating mask until Dice > tau.

        Args:
            model (torch.nn.Module): The segmentation model.
            X (torch.Tensor): Input image tensor.
            target_class (int): Target class to explain.
            max_dilations (int): Maximum number of dilations to try.

        Returns:
            Tuple containing:
                - X_SR: Masked image with sufficient region
                - M_SR: Sufficient region mask
                - n_dilations: Number of dilations performed
        """
        # Get original prediction
        with torch.no_grad():
            Y0 = generate_output_prediction(model, X)

        # Initialize mask with predicted object
        M = (Y0 == target_class).float()

        n_dilations = 0

        # Dilate until sufficient
        for n_dilations in range(max_dilations + 1):
            # Apply mask to image
            X_masked = generate_masked_image(X, M)

            # Get prediction on masked image
            with torch.no_grad():
                Y_masked = generate_output_prediction(model, X_masked)

            # Compute Dice score
            dice_score = evaluated_single_masked_prediction(
                Y0, Y_masked, target_class, self.score_mode, self.num_classes
            )

            # Check if sufficient
            if dice_score >= self.tau:
                break

            # Dilate mask for next iteration
            if n_dilations < max_dilations:
                M = dilate_mask(M, kernel_size=self.dilation_kernel_size)

        X_SR = generate_masked_image(X, M)
        return X_SR, M, n_dilations

    def optimize_mask(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        X_SR: torch.Tensor,
        M_SR: torch.Tensor,
        target_class: int,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Optimize mask to find minimally sufficient region (Stage 2).

        Args:
            model (torch.nn.Module): The segmentation model.
            X (torch.Tensor): Original input image.
            X_SR (torch.Tensor): Sufficient region masked image.
            M_SR (torch.Tensor): Sufficient region mask (initialization).
            target_class (int): Target class to explain.
            logger (Optional[logging.Logger]): Logger for tracking losses.

        Returns:
            Tuple containing:
                - Optimized minimally sufficient mask (M_MSR)
                - Loss history dictionary with keys: 'total', 'sparsity', 'tv', 'dice'
        """
        # Get original prediction for reference
        with torch.no_grad():
            Y0 = generate_output_prediction(model, X)

        # Initialize optimizable mask from sufficient region
        M = M_SR.clone().detach().requires_grad_(True)

        # Setup optimizer
        optimizer = torch.optim.AdamW([M], lr=self.learning_rate)

        # Initialize loss history
        loss_history = {
            "total": [],
            "sparsity": [],
            "tv": [],
            "dice": [],
        }

        # Optimization loop
        for iteration in range(self.n_iterations):
            optimizer.zero_grad()

            # Apply mask to sufficient region
            X_masked = X_SR * M

            # Get prediction on masked image
            Y_masked = generate_output_prediction(model, X_masked)

            # Compute losses
            # 1. Sparsity loss: encourage fewer pixels
            sparsity_loss = self.lambda_sparsity * M.abs().mean()

            # 2. Total variation loss: encourage smooth masks
            tv_loss = self.gamma_tv * total_variation_loss(M)

            # 3. Dice preservation loss: maintain prediction quality
            dice_loss = compute_weighted_dice_loss(
                Y_masked.squeeze(),
                Y0.squeeze(),
                target_class,
                alpha_fg=self.alpha_fg,
                alpha_bg=self.alpha_bg,
            )

            # Total loss
            total_loss = sparsity_loss + tv_loss + dice_loss

            # Store loss values
            loss_history["total"].append(total_loss.item())
            loss_history["sparsity"].append(sparsity_loss.item())
            loss_history["tv"].append(tv_loss.item())
            loss_history["dice"].append(dice_loss.item())

            # Log losses
            if logger is not None and iteration % 10 == 0:
                logger.info(
                    f"Iteration {iteration}/{self.n_iterations}: "
                    f"Total Loss={total_loss.item():.6f}, "
                    f"Dice Loss={dice_loss.item():.6f}, "
                    f"Sparsity Loss={sparsity_loss.item():.6f}, "
                    f"TV Loss={tv_loss.item():.6f}"
                )

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            # Clamp mask values: < 0.2 -> 0, clip to [0, 1]
            with torch.no_grad():
                M.data = torch.clamp(M.data, 0.0, 1.0)
                M.data[M.data < 0.2] = 0.0

        # Log final losses
        if logger is not None:
            logger.info(
                f"Final Iteration {self.n_iterations}: "
                f"Total Loss={loss_history['total'][-1]:.6f}, "
                f"Dice Loss={loss_history['dice'][-1]:.6f}, "
                f"Sparsity Loss={loss_history['sparsity'][-1]:.6f}, "
                f"TV Loss={loss_history['tv'][-1]:.6f}"
            )

        return M.detach(), loss_history

    def compute_saliency(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        target_class: int,
        return_details: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        """Compute MiSuRe saliency maps (full two-stage pipeline).

        Args:
            model (torch.nn.Module): The segmentation model.
            X (torch.Tensor): Input image tensor.
            target_class (int): Target class to explain.
            return_details (bool): Whether to return detailed outputs.
            logger (Optional[logging.Logger]): Logger for tracking optimization losses.

        Returns:
            Dictionary containing:
                - saliency_msr: Minimally sufficient region saliency map (primary output)
                - saliency_sr: Sufficient region saliency map (coarse)
                - mask_msr: Minimally sufficient region mask
                - mask_sr: Sufficient region mask
                - loss_history: Dictionary of loss values over iterations
                - n_dilations: Number of dilations performed (if return_details=True)
        """
        # Stage 1: Generate sufficient region
        if logger is not None:
            logger.info(f"Stage 1: Generating sufficient region for class {target_class}...")

        X_SR, M_SR, n_dilations = self.generate_sufficient_region(
            model, X, target_class
        )

        if logger is not None:
            logger.info(f"Stage 1 complete: {n_dilations} dilations performed")
            logger.info(f"Stage 2: Optimizing mask for minimally sufficient region...")

        # Stage 2: Optimize for minimally sufficient region
        M_MSR, loss_history = self.optimize_mask(
            model, X, X_SR, M_SR, target_class, logger
        )

        if logger is not None:
            logger.info(f"Stage 2 complete: Optimization finished")

        # Generate final saliency maps
        saliency_sr = M_SR
        saliency_msr = M_MSR

        # Prepare output
        result = {
            "saliency_msr": saliency_msr,
            "saliency_sr": saliency_sr,
            "mask_msr": M_MSR,
            "mask_sr": M_SR,
            "loss_history": loss_history,
        }

        if return_details:
            result["n_dilations"] = n_dilations
            result["X_SR"] = X_SR.detach().cpu()
            result["M_SR"] = M_SR.detach().cpu()
            result["X_MSR"] = (X_SR * M_MSR).detach().cpu()
            result["M_MSR"] = M_MSR.detach().cpu()

        return result
