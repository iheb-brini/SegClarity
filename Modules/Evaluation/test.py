import pytest
import torch

from Modules.Evaluation.core import CompletenessMetric, PointingGameMetric, FaithfulnessMetric


# ------------------------
# Fixtures: Mock Segmentation Data
# ------------------------
@pytest.fixture
def mock_segmentation_preds():
    """
    Mock logits for semantic segmentation (C=3 classes, H=4, W=4).
    Class 0 is target class in tests.
    """
    C, H, W = 3, 4, 4

    # Original prediction: class 0 has high logit values
    pred = torch.full((C, H, W), 1.0)
    pred[0] = 3.0  # target class more confident everywhere

    # Masked prediction: class 0 loses confidence in some pixels
    pred_on_masked = pred.clone()
    pred_on_masked[0, 1:3, 1:3] = 0.5  # drop confidence in central region

    return pred, pred_on_masked


@pytest.fixture
def mock_saliency_and_gt_mask():
    """
    Saliency map (H=4, W=4) and GT mask for class 0.
    Max saliency at (2,2) which is inside GT mask.
    """
    H, W = 4, 4
    saliency = torch.zeros((H, W))
    saliency[2, 2] = 0.95  # most salient pixel
    saliency[0, 0] = 0.5   # some other saliency

    gt_mask = torch.zeros((H, W), dtype=torch.int)
    gt_mask[1:4, 1:4] = 1  # square object in bottom-right

    return saliency, gt_mask


@pytest.fixture
def mock_faithfulness_data_seg():
    """
    Saliency and per-pixel drops for a segmentation map.
    Drops are proportional to saliency for positive correlation.
    """
    H, W = 4, 4
    saliency = torch.rand((H, W))
    drops = saliency * 0.8  # strong positive correlation

    return saliency, drops


# ------------------------
# CompletenessMetric Tests
# ------------------------
def test_completeness_metric_basic(mock_segmentation_preds):
    pred, pred_on_masked = mock_segmentation_preds
    metric = CompletenessMetric()
    score = metric.calculate(pred, pred_on_masked, target_class=0)

    assert score > 0, "Completeness should be positive when masking reduces target confidence"
    assert isinstance(score, float)
    assert 0 <= score <= 1, "Completeness should be normalized in [0,1] range"


def test_completeness_metric_no_drop(mock_segmentation_preds):
    pred, _ = mock_segmentation_preds
    metric = CompletenessMetric()
    score = metric.calculate(pred, pred, target_class=0)
    assert score == 0.0


def test_completeness_metric_negative_drop(mock_segmentation_preds):
    pred, pred_on_masked = mock_segmentation_preds
    metric = CompletenessMetric()
    # Swap arguments → masked output has higher confidence
    score = metric.calculate(pred_on_masked, pred, target_class=0)
    assert score == 0.0


# ------------------------
# PointingGameMetric Tests
# ------------------------
def test_pointing_game_hit(mock_saliency_and_gt_mask):
    saliency, gt_mask = mock_saliency_and_gt_mask
    metric = PointingGameMetric()
    score = metric.calculate(saliency, gt_mask)
    assert score == 1.0, "Most salient pixel is inside GT mask"


def test_pointing_game_miss(mock_saliency_and_gt_mask):
    saliency, gt_mask = mock_saliency_and_gt_mask
    metric = PointingGameMetric()
    # Move max saliency to a background pixel
    saliency[0, 0] = 1.0
    saliency[2, 2] = 0.1
    score = metric.calculate(saliency, gt_mask)
    assert score == 0.0, "Most salient pixel should be outside GT mask"


# ------------------------
# FaithfulnessMetric Tests
# ------------------------
def test_faithfulness_metric_positive(mock_faithfulness_data_seg):
    saliency, drops = mock_faithfulness_data_seg
    metric = FaithfulnessMetric()
    score = metric.calculate(saliency, drops)
    assert score > 0.5, "Correlation should be strongly positive"


def test_faithfulness_metric_zero_case():
    saliency = torch.zeros((4, 4))
    drops = torch.zeros((4, 4))
    metric = FaithfulnessMetric()
    score = metric.calculate(saliency, drops)
    assert score == 0.0


def test_faithfulness_metric_negative_corr():
    saliency = torch.tensor([
        [0.1, 0.9, 0.2, 0.8],
        [0.8, 0.2, 0.9, 0.1],
        [0.3, 0.7, 0.4, 0.6],
        [0.6, 0.4, 0.7, 0.3]
    ])
    drops = 1.0 - saliency  # perfect inverse correlation
    metric = FaithfulnessMetric()
    score = metric.calculate(saliency, drops)
    assert score < -0.9, "Should be strongly negative"
