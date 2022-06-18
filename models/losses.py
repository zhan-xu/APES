import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F


def infoNCE(feature1, feature2, corr, tau):
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss = 0.0
    for i in range(len(feature1)):
        feature1_i = feature1[i]
        feature2_i = feature2[i]
        corr_i = corr[i]

        pts_from_all = torch.where(corr_i[:, :, 0] > -1)

        # randomly choose a subset of correspondences for training
        sample_ids = np.random.choice(np.arange(len(pts_from_all[0])), min(len(pts_from_all[0]), 1024), replace=False)
        pts_from = [pts_from_all[0][sample_ids], pts_from_all[1][sample_ids]]
        pts_from = torch.stack(pts_from, dim=1)

        pts_to = corr_i[pts_from[:, 0], pts_from[:, 1], 0:2].long()

        pf_from = feature1_i[:, pts_from[:, 0], pts_from[:, 1]].T
        pf_to = feature2_i[:, pts_to[:, 0], pts_to[:, 1]].T

        label_i = torch.randperm(len(pf_to)).to(corr_i.device)
        pf_from_perm = pf_from[label_i]

        prod = torch.mm(pf_from_perm, pf_to.T) / tau
        loss_i = cross_entropy_loss(prod, label_i)
        loss += loss_i.mean()
    return loss / len(feature1)


def hungarian_matching(pred_seg, gt_seg):
    intersect = (np.expand_dims(pred_seg, -1) * np.expand_dims(gt_seg, -2)).sum(axis=0).sum(axis=0)
    matching_cost = 1-np.divide(intersect, pred_seg.sum(0).sum(0)[:, None]+gt_seg.sum(0).sum(0)[None, :]-intersect+1e-8)
    row_ind, col_ind = linear_sum_assignment(matching_cost)
    return np.vstack((row_ind, col_ind))


def hungarian_matching_slic(pred_seg, gt_seg):
    interset = np.matmul(pred_seg.T, gt_seg)
    matching_cost = 1 - np.divide(interset, np.expand_dims(np.sum(pred_seg, 0), 1) + np.sum(gt_seg, axis=0,  keepdims=True) - interset + 1e-8)
    row_ind, col_ind = linear_sum_assignment(matching_cost)
    return np.vstack((row_ind, col_ind))


def motionLoss(trans_diff, gt_seg):
    """ trans_diff: [B, 2, S, S] ,
        gt_seg: [B, S, S] """
    diff_masked = trans_diff * gt_seg.unsqueeze(dim=1)
    loss = torch.sum(diff_masked ** 2, dim=1).sum() / (2 * gt_seg.sum())
    return loss / len(trans_diff)


def groupingLoss(sim_mats_pred, sim_mats_gt):
    """ sim_mats_pred: [B, S, S]
        sim_mats_gt: [B, S, S] """
    loss = torch.nn.functional.binary_cross_entropy_with_logits(sim_mats_pred, sim_mats_gt.float())
    return loss


def iouLoss(seg_pred, seg_gt):
    """
    seg_pred: [B, H, W, S], 0-1, soft probability in dim=-1, seg_pred.sum(-1)=1.0
    seg_gt: [B, H, W, S], 0 or 1, one-hot in dim=-1
    """
    loss = 0.0
    for i in range(len(seg_pred)):
        pred_seg_i = seg_pred[i].float()
        gt_seg_i = seg_gt[i]
        matching_id_i = hungarian_matching(pred_seg_i.detach().cpu().numpy(), gt_seg_i.detach().cpu().numpy())
        pred_seg_i_reorder = pred_seg_i[..., matching_id_i[0]]
        gt_seg_i_reorder = gt_seg_i[..., matching_id_i[1]]
        interset = (pred_seg_i_reorder * gt_seg_i_reorder).sum(0).sum(0)
        cost_i = 1 - torch.div(interset, pred_seg_i_reorder.sum(0).sum(0) + gt_seg_i_reorder.sum(0).sum(0) - interset + 1e-8)
        loss = loss + cost_i.mean()
    return loss / len(seg_pred)


def iouLoss_slic(pred_seg, gt_seg):
    """
    seg_pred: [B, NUM_SLIC, TRAIN_S],
    seg_gt: [B, NUM_SLIC, GT_S]
    """
    loss = 0.0
    for i in range(len(pred_seg)):
        pred_seg_i = pred_seg[i]
        gt_seg_i = gt_seg[i]
        matching_id_i = hungarian_matching_slic(pred_seg_i.detach().cpu().numpy(), gt_seg_i.detach().cpu().numpy())
        pred_seg_i_reorder = pred_seg_i[:, matching_id_i[0]]
        gt_seg_i_reorder = gt_seg_i[:, matching_id_i[1]]
        interset = torch.sum(pred_seg_i_reorder * gt_seg_i_reorder, dim=0)
        cost_i = 1 - torch.div(interset, pred_seg_i_reorder.sum(dim=0) + gt_seg_i_reorder.sum(dim=0) - interset + 1e-8)
        loss = loss + cost_i.mean()
    return loss / len(pred_seg)