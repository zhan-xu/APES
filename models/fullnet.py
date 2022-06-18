import torch
import torch.nn as nn
if torch.cuda.is_available():
    import torch_cluster.knn as knn
else:
    raise Exception('No GPU detected, need an alternative knn function.')
from torch_scatter import scatter_add
from models.corrnet import CorrNet
from models.clusternet import ClusterNet
import numpy as np, cv2, os
from utils.vis_utils import visualize_corr

__all__ = ['fullnet']


def make_coordinate_grid2(spatial_size, type):
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    meshed = torch.cat([yy.unsqueeze_(-1), xx.unsqueeze_(-1)], -1)
    return meshed


def form_pred_corr_only(img1_feature, img2_feature, mask1, mask2, tau):
    masked_pos_src = torch.nonzero(mask1)
    masked_pos_dst = torch.nonzero(mask2)
    feat_x = img1_feature[masked_pos_src[:, 0], :, masked_pos_src[:, 1], masked_pos_src[:, 2]]
    feat_y = img2_feature[masked_pos_dst[:, 0], :, masked_pos_dst[:, 1], masked_pos_dst[:, 2]]
    batch_x = masked_pos_src[:, 0]
    batch_y = masked_pos_dst[:, 0]
    with torch.no_grad():
        assign_index_xy = knn(feat_y, feat_x, 3, batch_y, batch_x, cosine=True)

    #sim_xy = torch.sum(feat_x[assign_index_xy[0]] * feat_y[assign_index_xy[1]], dim=-1, keepdim=True) / tau
    #sim_xy = torch.exp(sim_xy - sim_xy.max())
    #pred_target_pos_forward = scatter_add(masked_pos_dst[assign_index_xy[1]] * sim_xy, assign_index_xy[0], dim=0) / scatter_add(sim_xy, assign_index_xy[0], dim=0)

    # In practice, we find the following simplification is more efficient. sim_xy is always postive since we only consider top-3 most similar pairs.
    sim_xy = torch.sum(feat_x[assign_index_xy[0]] * feat_y[assign_index_xy[1]], dim=-1, keepdim=True)
    pred_target_pos_forward = scatter_add(masked_pos_dst[assign_index_xy[1]] * sim_xy, assign_index_xy[0], dim=0) / scatter_add(sim_xy, assign_index_xy[0], dim=0)

    pred_corr_forward = -torch.ones((mask1.shape[0], mask1.shape[1], mask1.shape[2], 2), device=mask1.device)
    pred_corr_forward[masked_pos_src[:, 0], masked_pos_src[:, 1], masked_pos_src[:, 2], :] = pred_target_pos_forward[:, 1:]
    return pred_corr_forward


class FullNet(nn.Module):
    def __init__(self, n_channels, n_classes, offline_corr, bilinear=True, train_s=8):
        super(FullNet, self).__init__()
        self.offline_corr = offline_corr
        self.corrnet = CorrNet(n_channels, n_classes, bilinear)
        self.clusternet = ClusterNet(train_s=train_s)

    def forward(self, img_1, img_2, mask_1, mask_2, img_grid, slic, src_pixel_group, pred_corr=None):
        """
        :param img_1, img_2:     B, 3, H, W
        :param mask_1, mask_2:   B, H, W
        :param img_grid:         B, 2, H, W
        :param slic:             B, H, W
        :param src_pixel_group:  B, SLIC, SAMPLE, 2 (B, 50, 120, 2)
        :param pred_corr:          B, H, W, 2
        :return:
        """
        if self.offline_corr:
            img1_feature, img2_feature, tau = None, None, None
            pred_corr_onthefly = pred_corr.float()
        else:  # train with pred corr
            img1_feature, img2_feature, tau = self.corrnet(img_1, img_2, mask_1.unsqueeze(1), mask_2.unsqueeze(1))
            pred_corr_onthefly = form_pred_corr_only(img1_feature, img2_feature, mask_1, mask_2, tau)

        # debug
        # for i in range(len(img_1)):
        #     pred_corr_i = pred_corr_onthefly[i].detach().to("cpu").numpy()
        #     img_1_np = img_1[i].to("cpu").numpy().transpose(1, 2, 0)
        #     img_2_np = img_2[i].to("cpu").numpy().transpose(1, 2, 0)
        #     img_1_np = np.round(img_1_np * 255).astype(np.uint8)
        #     img_2_np = np.round(img_2_np * 255).astype(np.uint8)
        #     mask_1_np = np.round(mask_1[i].to("cpu").numpy() * 255).astype(np.uint8)
        #     img_corr = visualize_corr(img_1_np, mask_1_np, img_2_np, pred_corr_i, show=False)
        #     cv2.imshow("pred_corr", img_corr)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()

        dst_pixel_group = []
        for i in range(src_pixel_group.shape[0]):
            dst_pixel_group_i = []
            for g in range(src_pixel_group.shape[1]):
                src_pixel_group_g = torch.round(src_pixel_group[i, g] * 256).long()
                src_pixel_group_g = torch.clamp(src_pixel_group_g, 0, 255)
                dst_pixel_group_g = pred_corr_onthefly[i, src_pixel_group_g[:, 0], src_pixel_group_g[:, 1]]
                dst_pixel_group_i.append(dst_pixel_group_g)
            dst_pixel_group.append(torch.stack(dst_pixel_group_i, dim=0))
        dst_pixel_group = torch.stack(dst_pixel_group, dim=0)
        dst_pixel_group /= (slic.shape[1])

        pos_tar = pred_corr_onthefly.permute((0, 3, 1, 2)) / pred_corr_onthefly.shape[1]
        D, A, S_slic, pred_R, pred_T = self.clusternet(img_grid, pos_tar, mask_1.unsqueeze(1), slic, src_pixel_group, dst_pixel_group)

        return {"img1_feature": img1_feature, "img2_feature": img2_feature, "tau": tau, "pred_corr": pred_corr_onthefly,
                "D": D, "A": A, "S": S_slic, "pred_R": pred_R, "pred_T": pred_T}

    def slic(self, corr, slic, src_pixel_group):
        """
        corr: B, H, W, 2
        slic: B, H, W
        src_pixel_group: B, S, G, 2
        """
        dst_pixel_group = []
        for i in range(src_pixel_group.shape[0]):
            dst_pixel_group_i = []
            for g in range(src_pixel_group.shape[1]):
                src_pixel_group_g = torch.round(src_pixel_group[i, g] * 256).long()
                src_pixel_group_g = torch.clamp(src_pixel_group_g, 0, 255)
                dst_pixel_group_g = corr[i, src_pixel_group_g[:, 0], src_pixel_group_g[:, 1]]
                dst_pixel_group_i.append(dst_pixel_group_g)
            dst_pixel_group.append(torch.stack(dst_pixel_group_i, dim=0))
        dst_pixel_group = torch.stack(dst_pixel_group, dim=0)
        dst_pixel_group /= (slic.shape[1])
        return dst_pixel_group

    def test_run_one_direction(self, img, mask, pred_corr, slic, src_pixel_group):
        dst_pixel_group = self.slic(pred_corr, slic, src_pixel_group)
        pos_tar = pred_corr.permute((0, 3, 1, 2)) / pred_corr.shape[1]
        img_grid = make_coordinate_grid2((mask.shape[1], mask.shape[2]), type=torch.float32).to(img.device)
        img_grid = img_grid[None, ...].repeat(img.shape[0], 1, 1, 1).to(img.device)
        img_grid = img_grid / torch.FloatTensor([[[[img.shape[-2], img.shape[-1]]]]]).to(img_grid.device)
        D, A, S, R, T = self.clusternet(img_grid.permute(0, 3, 1, 2), pos_tar, mask.float().unsqueeze(1), slic, src_pixel_group, dst_pixel_group)
        return D, A, S, R, T, slic

    def test_run(self, img_1, img_2, mask_1, mask_2, slic_1, slic_2, src_pixel_group_1, src_pixel_group_2, pair_name=None, debug_folder=None):
        img1_feature, img2_feature, tau = self.corrnet(img_1, img_2, mask_1.unsqueeze(1), mask_2.unsqueeze(1))
        pred_corr_forward = form_pred_corr_only(img1_feature, img2_feature, mask_1, mask_2, tau)
        pred_corr_backward = form_pred_corr_only(img2_feature, img1_feature, mask_2, mask_1, tau)

        if False:  # visualzation for debug
            for i in range(len(pred_corr_forward)):
                pair = (pair_name[0][i].item(), pair_name[1][i].item())
                img_1_np = img_1[i].to("cpu").numpy().transpose(1, 2, 0)
                img_2_np = img_2[i].to("cpu").numpy().transpose(1, 2, 0)
                img_1_np = np.round(img_1_np * 255).astype(np.uint8)
                img_2_np = np.round(img_2_np * 255).astype(np.uint8)
                mask_1_np = np.round(mask_1[i].to("cpu").numpy() * 255).astype(np.uint8)
                mask_2_np = np.round(mask_2[i].to("cpu").numpy() * 255).astype(np.uint8)

                pred_corr_forward_i = pred_corr_forward[i].detach().to("cpu").numpy()
                img_corr = visualize_corr(img_1_np, mask_1_np, img_2_np, pred_corr_forward_i, show=True)
                #cv2.imwrite(os.path.join(debug_folder, f"corr_{pair[0]}{pair[1]}.png"), img_corr)

                pred_corr_backward_i = pred_corr_backward[i].detach().to("cpu").numpy()
                img_corr = visualize_corr(img_2_np, mask_2_np, img_1_np, pred_corr_backward_i, show=True)
                #cv2.imwrite(os.path.join(debug_folder, f"corr_{pair[1]}{pair[0]}.png"), img_corr)

        D_f, A_f, S_f, R_f, T_f, slic_f = \
            self.test_run_one_direction(img_1, mask_1, pred_corr_forward, slic_1, src_pixel_group_1)
        D_b, A_b, S_b, R_b, T_b, slic_b = \
            self.test_run_one_direction(img_2, mask_2, pred_corr_backward, slic_2, src_pixel_group_2)
        return {"pred_corr": torch.cat((pred_corr_forward, pred_corr_backward), dim=-1),
                "slic_f": slic_f, "slic_b": slic_b,
                "D_f": D_f, "A_f": A_f, "S_f": S_f, "R_f": R_f, "T_f": T_f,
                "D_b": D_b, "A_b": A_b, "S_b": S_b, "R_b": R_b, "T_b": T_b}


def fullnet(**kwargs):
    model = FullNet(n_channels=kwargs["n_channels"], n_classes=kwargs["n_classes"], bilinear=kwargs["bilinear"],
                    train_s=kwargs["train_s"], offline_corr=kwargs["offline_corr"])
    return model
