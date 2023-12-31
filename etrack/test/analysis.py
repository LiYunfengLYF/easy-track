import torch
import numpy as np
from ..utils import txtread


def calc_precision(pred_bb, anno_bb, normalized=False):
    pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
    anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)

    if normalized:
        pred_center = pred_center / anno_bb[:, 2:]
        anno_center = anno_center / anno_bb[:, 2:]

    err_center = ((pred_center - anno_center) ** 2).sum(1).sqrt()
    return err_center


def calc_iou(pred_bb, anno_bb):
    tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
    br = torch.min(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = (br - tl + 1.0).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection

    return (intersection / union)


def calc_curves(ious, center_errors, norm_center_errors):
    ious = np.asarray(ious, float)[:, np.newaxis]

    center_errors = np.asarray(center_errors, float)[:, np.newaxis]
    norm_center_errors = np.asarray(norm_center_errors, float)[:, np.newaxis]

    thr_iou = np.linspace(0, 1, 21)[np.newaxis, :]
    thr_ce = np.arange(0, 51)[np.newaxis, :]
    thr_ce_norm = np.arange(0, 51)[np.newaxis, :] / 100.0

    bin_iou = np.greater(ious, thr_iou)
    bin_ce = np.less_equal(center_errors, thr_ce)
    bin_norm_ce = np.less_equal(norm_center_errors, thr_ce_norm)

    succ_curve = np.mean(bin_iou, axis=0)
    prec_curve = np.mean(bin_ce, axis=0)
    norm_prec_curve = np.mean(bin_norm_ce, axis=0)

    return succ_curve, prec_curve, norm_prec_curve


def calc_seq_performace(results_boxes, gt_boxes):
    assert len(results_boxes) == len(gt_boxes)
    results_boxes = torch.tensor(results_boxes)
    gt_boxes = torch.tensor(gt_boxes)

    center_errors = calc_precision(results_boxes, gt_boxes)
    norm_enter_errors = calc_precision(results_boxes, gt_boxes, normalized=True)
    ious = calc_iou(results_boxes, gt_boxes)

    succ_curve, prec_curve, norm_prec_curve = calc_curves(ious, center_errors, norm_enter_errors)

    auc_score = np.mean(succ_curve)
    prec_score = prec_curve[20]
    norm_prec_score = norm_prec_curve[20]

    return auc_score, prec_score, norm_prec_score


def report_seq_performance(gt_file: str, results_file: str) -> None:
    """
    Description
        calc Success Score, Precision Score, Norm Precision Score, Success Rate of on a sequence

    Params:
        gt_file:        str
        results_file:   str

    """

    results_boxes = txtread(results_file)
    gt_boxes = txtread(gt_file)

    succ_score, prec_score, norm_prec_score = calc_seq_performace(results_boxes, gt_boxes)
    print(f'Performance: ')
    print(f'\tAUC:\t\t\t\t{round(succ_score, 2)}')
    print(f'\tPrecision Score:\t\t\t{round(prec_score, 2)}')
    print(f'\tNorm Precision Score:\t\t{round(norm_prec_score, 2)}')
