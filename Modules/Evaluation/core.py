"""
Original implementation is found in https://github.com/aimagelab/ADCC
"""

from abc import ABC, abstractmethod
from typing import Callable, List
from .tools import isNaN, pcc,pcc_cuda,jaccard_index,_fast_hist
from torch.nn.functional import softmax
from torch import clip
import torch
from captum.metrics import infidelity,sensitivity_max
from Modules.Attribution import generateAttributions


class Metric(ABC):
    name: str
    func: Callable

    def __init__(self, name) -> None:
        self.name = name

    @abstractmethod
    def calculate(self, *args, **kwargs) -> float:
        pass

    def __repr__(self) -> str:
        return f"{self.name}"

    def __str__(self) -> str:
        return f"{self.name}"

class ComplexityMetric(Metric):
    def __init__(self, normalize_map) -> None:
        self.normalize_map = normalize_map
        super().__init__('complexity')

    def calculate(self, saliency_map) -> float:
        if saliency_map.sum() == 0:
            return 0
        com = float(self.normalize_map(abs(saliency_map)).sum().item() /
                    (saliency_map.shape[-1]*saliency_map.shape[-2]))
        if isNaN(com):
            com = 1.0
        return com


class CoherencyMetric(Metric):
    def __init__(self) -> None:
        super().__init__('coherency')

    def calculate(self, saliency_map_A, saliency_map_B) -> float:
        # Pearson correlation coefficient
        y = pcc_cuda(saliency_map_A, saliency_map_B)
        return y

class AverageDropIoUMetric(Metric):
    def __init__(self, num_classes) -> None:
        super().__init__('avgdropIoU')
        self.num_classes = num_classes

    def calculate(self, out, out_on_exp, target) -> float:
        out = out.detach().cpu()
        out_on_exp = out_on_exp.detach().cpu()

        hist = _fast_hist(out,
                          out_on_exp,
                          num_classes=self.num_classes)
        _, jacc = jaccard_index(hist)
        return 1-jacc[target].item()

class AverageDropMetric(Metric):
    def __init__(self) -> None:
        super().__init__('avgdrop')

    def calculate(self, pred, pred_on_exp, target) -> float:
        if len(pred.shape) > 3:
            pred = pred.squeeze()
            pred_on_exp = pred_on_exp.squeeze()
            
        h, w = pred.shape[-2:]
        A = softmax(pred, dim=0)[target, :, :]
        B = softmax(pred_on_exp, dim=0)[target, :, :]
        return (clip(A - B, min=0) / A).sum().item() / (h*w)
    

class ADCCMetric(Metric):
    def __init__(self) -> None:
        super().__init__('adcc')

    def calculate(self, com, avgdrop, coh) -> float:
      try:
        adcc= 3 / (1/coh + 1/(1-com) + 1/(1-avgdrop))
      except:
        adcc= 0.0

      return adcc
class CompletenessMetric(Metric):
    def __init__(self) -> None:
        super().__init__('completeness')

    def calculate(self, pred: torch.Tensor, pred_on_masked: torch.Tensor, target_class: int) -> float:
        prob_orig = torch.softmax(pred, dim=0)[target_class, :, :]
        prob_masked = torch.softmax(pred_on_masked, dim=0)[target_class, :, :]
        avg_orig = prob_orig.mean().item()
        avg_masked = prob_masked.mean().item()
        completeness = max(0.0, avg_orig - avg_masked)
        return completeness


class PointingGameMetric(Metric):
    def __init__(self) -> None:
        super().__init__('pointing_game')

    def calculate(self, saliency_map: torch.Tensor, gt_mask: torch.Tensor) -> float:
        max_pos = torch.argmax(saliency_map)
        h, w = saliency_map.shape
        y, x = divmod(max_pos.item(), w)
        return float(gt_mask[y, x] > 0)


class FaithfulnessMetric(Metric):
    def __init__(self) -> None:
        super().__init__('faithfulness')

    def calculate(self, saliency_map: torch.Tensor, drops: torch.Tensor) -> float:
        saliency_flat = saliency_map.flatten()
        drops_flat = drops.flatten()

        if saliency_flat.numel() == 0 or drops_flat.numel() == 0:
            return 0.0

        corr_matrix = torch.corrcoef(torch.stack([saliency_flat, drops_flat]))
        corr = corr_matrix[0, 1].item()
        if torch.isnan(torch.tensor(corr)):
            corr = 0.0
        return corr

###############################
from Modules.Utils import Normalizations
from captum.attr import LayerAttribution

normalizer_no_shift = Normalizations.pick(Normalizations.normalize_non_shifting_zero)
normalizer_0_1 = Normalizations.pick(Normalizations.normalize_0_1)
normalizer_scaler = Normalizations.pick(Normalizations.normalize_scaler)


###################
# Content heatmap #
###################

def compute_content_heatmap(attr,mask,target,normalizer=normalizer_scaler):
    attr_normalized = normalizer(attr)
    if attr_normalized.shape != mask.shape:
        attr_normalized = LayerAttribution.interpolate(
            attr_normalized, mask.squeeze().shape[-2:]
        )
    V = attr_normalized[0][mask==target]
    V = normalizer_0_1(V)
    if len(V) == 0:
        return 0
    
    return V.sum().item() / (len(V))


##########################
# Attribution Similarity #
##########################

def f_sum_attribution_advanced(V, sign="pos", **kwargs):
    V_pos = V[V > 0]
    V_pos_sum = V_pos.sum().item()
    V_pos_len = len(V_pos)

    V_neg = V[V < 0]
    V_neg_sum = V_neg.sum().item()
    V_neg_len = len(V_neg)

    V_zero = V[V == 0]
    V_zero_len = len(V_zero)

    if sign == "pos":
        score = V_pos_sum + V_neg_sum
    elif sign == "neg":
        score = -(V_pos_sum + V_neg_sum)
    else:
        raise Exception("Not implemented")

    return score, {
        "pos": round(V_pos_sum, 2),
        "neg": round(V_neg_sum, 2),
        "%pos": round((V_pos_len / (len(V))) * 100, 2),
        "%neg": round((V_neg_len / (len(V))) * 100, 2),
        "%zero": round((V_zero_len / (len(V))) * 100, 2),
    }


def get_component_info(attr, mask, c, sign="pos"):
    if "pos" == sign:
        V = attr[mask == c]
    else:
        V = attr[mask != c]

    if len(V) == 0:
        return 0
    return f_sum_attribution_advanced(V, sign)


def compute_ACS(attr, mask, target):
    attr_normalized = normalizer_no_shift(attr)

    if attr_normalized.shape != mask.shape:
        attr_normalized = LayerAttribution.interpolate(
            attr_normalized, mask.squeeze().shape[-2:]
        )

    w_target = round(
        len(mask[mask == target]) / (mask.shape[-1] * mask.shape[-2]) * 100, 2
    )

    # target score
    score_target, info_target = get_component_info(
        attr_normalized[0], mask, c=target, sign="pos"
    )

    TP, FP = info_target["pos"], abs(info_target["neg"])

    # non target score
    score_non_target, info_non_target = get_component_info(
        attr_normalized[0], mask, c=target, sign="neg"
    )

    TN, FN = abs(info_non_target["neg"]), info_non_target["pos"]

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f1 = 2 * precision * recall / (precision + recall)

    metrics = {"accuracy":accuracy,"precision": precision, "recall": recall, "f1": f1}

    details = {
        "w_target":w_target,
        "pos_target":info_target['pos'],
        "neg_target":info_target['neg'],
        "pos_non_target":info_non_target['pos'],
        "neg_non_target":info_non_target['neg'],
        "score_target":score_target,
        "score_non_target":score_non_target,
        "acccuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1,
    }
    return metrics, details

###################
# Sensitivity Max #
###################


class SensitivityMax:
    def __init__(self, n_perturb_samples=5,
                perturb_radius=0.02,
                max_examples_per_batch=1) -> None:
        self.n_perturb_samples = n_perturb_samples
        self.perturb_radius = perturb_radius
        self.max_examples_per_batch=max_examples_per_batch

    def explanation_func(self,x, model, method, layer, **kwargs):
        target = kwargs.get('target',None)
        return generateAttributions(
            x[0], model, target=target, method=method, layer=layer
        )

    def compute(self,x,model,method,layer,target):
        return sensitivity_max(
                explanation_func=self.explanation_func,
                inputs=x,
                target=target,
                n_perturb_samples=self.n_perturb_samples,
                perturb_radius=self.perturb_radius,
                model=model,
                method=method,
                layer=layer,
                max_examples_per_batch=self.max_examples_per_batch
        )