from Modules.Evaluation.core import DiceMetric, Metric
from Modules.Evaluation.tools import _fast_hist, jaccard_index

import torch
from typing import Literal
from torch import Tensor
from Modules.Ablation.tools import generate_mask_from_grid
from typing import Tuple
from tqdm import tqdm


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
