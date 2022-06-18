import torch
import torch.nn as nn
from models.corrnet import UNet_gated
from models.nn_util import Seq
from torch_scatter import scatter_mean
import numpy as np

__all__ = ["clustergnet"]

BN_CONFIG = {"class": "BatchNorm"}


class TransNet(nn.Module):
    def __init__(self, bn):
        super(TransNet, self).__init__()
        self.unet_r = UNet_gated(chn_in=5, chn_mid=[16, 32, 64, 128, 256], chn_out=16, downsample_time=4)
        self.mreg_r = Seq(16).conv1d(64, bn=bn).conv1d(2, activation=None)
        self.unet_t = UNet_gated(chn_in=5, chn_mid=[16, 32, 64, 128, 256], chn_out=16, downsample_time=4)
        self.mreg_t = Seq(16).conv1d(64, bn=bn).conv1d(2, activation=None)

    def forward(self, src_pos, tar_pos, mask_pixel, src_overseg):
        """
        :param src_pos: (B, 2, H, W)
        :param tar_pos: (B, 2, H, W)
        :param mask_pixel: (B, 1, H, W)
        :param src_overseg: (B, H, W)
        """
        # put invalid pixels to the last super pixel
        fake_seg = src_overseg.max() + 1
        src_overseg_no_negtive = src_overseg.clone()
        src_overseg_no_negtive[torch.any(tar_pos < 0, dim=1)] = fake_seg
        src_overseg_no_negtive[src_overseg_no_negtive < 0] = fake_seg
        src_overseg_no_negtive = src_overseg_no_negtive.reshape(src_overseg_no_negtive.shape[0], 1, -1).long()

        centroid_src = scatter_mean(src_pos.reshape(src_pos.shape[0], src_pos.shape[1], -1), src_overseg_no_negtive)
        centroid_src_pixel = torch.gather(centroid_src, dim=2, index=src_overseg_no_negtive.repeat(1, 2, 1))
        centroid_src_pixel = centroid_src_pixel.reshape(centroid_src.shape[0], 2, src_overseg.shape[1], src_overseg.shape[2])
        centroid_tar = scatter_mean(tar_pos.reshape(tar_pos.shape[0], tar_pos.shape[1], -1), src_overseg_no_negtive)
        centroid_tar_pixel = torch.gather(centroid_tar, dim=2, index=src_overseg_no_negtive.repeat(1, 2, 1))
        centroid_tar_pixel = centroid_tar_pixel.reshape(centroid_tar_pixel.shape[0], 2, src_overseg.shape[1], src_overseg.shape[2])

        src_pos_centered = src_pos - centroid_src_pixel
        dst_pos_centered = tar_pos - centroid_tar_pixel
        pos_map_0 = torch.cat((src_pos_centered, dst_pos_centered), dim=1)
        pos_map_0 = torch.where(torch.all(tar_pos >= 0, dim=1, keepdim=True), pos_map_0, -torch.ones_like(pos_map_0))

        x_rot_feature = self.unet_r(pos_map_0, mask_pixel.float())  # B, C, H, W
        x_rot_feature = x_rot_feature.reshape(x_rot_feature.shape[0], x_rot_feature.shape[1], -1)  # B, C, P
        pred_R = scatter_mean(x_rot_feature, src_overseg_no_negtive)
        pred_R = self.mreg_r(pred_R).permute(0, 2, 1)
        pred_R_mat = torch.stack((pred_R, torch.matmul(pred_R, torch.FloatTensor([[[0, 1], [-1, 0]]]).to(pred_R.device))), dim=-1)
        identity_mat = torch.eye(2, dtype=torch.float32, device=pred_R.device)
        pred_R_mat = pred_R_mat + identity_mat.unsqueeze(0).unsqueeze(0)

        pred_R_pixel = torch.gather(pred_R_mat, dim=1, index=src_overseg_no_negtive.permute(0, 2, 1).unsqueeze(-1).repeat(1, 1, 2, 2))
        pred_R_pixel = pred_R_pixel.reshape(pred_R_pixel.shape[0], src_overseg.shape[1], src_overseg.shape[2], 2, 2)
        src_pos_rot = torch.matmul(src_pos.permute(0, 2, 3, 1).unsqueeze(-2), pred_R_pixel).squeeze(-2).permute(0, 3, 1, 2)
        pos_map_1 = torch.cat((src_pos_rot, tar_pos), dim=1)
        pos_map_1 = torch.where(torch.all(tar_pos >= 0, dim=1, keepdim=True), pos_map_1, -torch.ones_like(pos_map_1))

        x_trans_feature = self.unet_t(pos_map_1, mask_pixel.float())  # B, C, H, W
        x_trans_feature = x_trans_feature.reshape(x_trans_feature.shape[0], x_trans_feature.shape[1], -1)  # B, C, P
        pred_T = scatter_mean(x_trans_feature, src_overseg_no_negtive)  # remove the last item
        pred_T = self.mreg_t(pred_T).permute(0, 2, 1)
        return pred_R_mat[:, :-1, :, :], pred_T[:, :-1, :].unsqueeze(dim=-2)


def sync_motion_seg(z_mat: torch.Tensor, force_d: int = -1, t: float = -np.inf, cut_thres: float = 0.1):
    e, V = torch.symeig(z_mat, eigenvectors=True)
    if force_d != -1:
        d = force_d
    else:
        assert z_mat.size(0) == 1
        e_leading = e[:, -10:]
        total_est_points = torch.sum(e_leading, -1)
        e_th = total_est_points * cut_thres
        e_count = torch.sum(e.detach() > e_th, dim=1)
        d = e_count.max().item()

    V = V[..., -d:]  # (B, M, d)
    e = e[..., -d:]  # (B, d)
    V = V * e.sqrt().unsqueeze(1)
    v_sign = V.detach().sum(dim=1, keepdim=True).sign()
    V = V * v_sign
    if t > -1e5:
        V[V < t] = 0.0

    return V


class VerifyNet(nn.Module):
    def __init__(self, bn, train_s):
        super(VerifyNet, self).__init__()
        self.u_pre_trans = Seq(4).conv2d(16, bn=bn).conv2d(64, bn=bn).conv2d(512, bn=bn)
        self.u_global_trans = Seq(512).conv2d(256, bn=bn).conv2d(256, bn=bn).conv2d(128, bn=bn)
        self.u_post_trans = Seq(512 + 256).conv2d(256, bn=bn).conv2d(64, bn=bn) \
            .conv2d(16, bn=bn).conv2d(1, activation=None)
        self.train_s = train_s

    def forward(self, src_pixel_group, dst_pixel_group, src_pos, pred_R, pred_T, overseg):
        """
        :param src_pixel_group/dst_pixel_group: (B, N_SUPERPIX, GROUP_SIZE, 2)
        :param src_pos: (B, 2, H, W)
        :param pred_R: (B, N_SUPERPIX, 2, 2)
        :param pred_T: (B, N_SUPERPIX, 1, 2)
        :param overseg: (B, H, W)
        """

        # get predicted and GT similarity matrices
        Rs = pred_R.unsqueeze(dim=1)
        ts = pred_T.unsqueeze(dim=1)
        diff = (torch.matmul(src_pixel_group.unsqueeze(dim=2), Rs.transpose(-2, -1)) + ts - dst_pixel_group.unsqueeze(dim=2)).mean(dim=-2)
        diff = diff + diff.transpose(2, 1)
        diff = diff.permute(0, 3, 1, 2)

        # get mean position of super pixels
        n_superpix = src_pixel_group.shape[1]
        src_overseg_no_negtive = overseg.clone()
        src_overseg_no_negtive[overseg == -1] = overseg.max() + 1
        pos_superpix = scatter_mean(
            src_pos.reshape(src_pos.shape[0], src_pos.shape[1], -1),
            src_overseg_no_negtive.reshape(src_overseg_no_negtive.shape[0], 1, -1).long(), dim=-1)[...,:-1]  # (2, N_SUPERPIX)

        # pass through networks
        U = torch.cat([diff, pos_superpix.unsqueeze(dim=-1).expand(-1, -1, -1, n_superpix)], dim=1)  # (B, 4, S, S)
        U = self.u_pre_trans(U)
        U_global0, _ = U.max(3, keepdim=True)
        U_global0 = self.u_global_trans(U_global0)
        U_global1, _ = U.max(2, keepdim=True)
        U_global1 = self.u_global_trans(U_global1)
        U = torch.cat([U, U_global0.expand(-1, -1, -1, n_superpix), U_global1.expand(-1, -1, n_superpix, -1)], dim=1)
        U = self.u_post_trans(U)
        pred_sim_before_sigmoid = U.squeeze(dim=1) # B, n_superpix, n_superpix

        # convert U to image space
        pred_sim = torch.sigmoid(pred_sim_before_sigmoid)
        pred_seg_slic_before_softmax = sync_motion_seg(pred_sim, force_d=self.train_s, t=0.0)  # B, n_superpix, self.train_s
        pred_seg_slic = torch.softmax(pred_seg_slic_before_softmax, dim=-1)
        return diff, pred_sim_before_sigmoid, pred_seg_slic


class ClusterNet(nn.Module):
    def __init__(self, train_s):
        super(ClusterNet, self).__init__()
        self.trans_net = TransNet(BN_CONFIG)
        self.verify_net = VerifyNet(BN_CONFIG, train_s)

    def forward(self, pos_src, pos_tar, mask, slic_map, src_pixel_group, dst_pixel_group):
        pred_R, pred_T = self.trans_net(pos_src, pos_tar, mask, slic_map)
        diff_superpix, sim_superpix, seg_slic = \
            self.verify_net(src_pixel_group, dst_pixel_group, pos_src, pred_R=pred_R, pred_T=pred_T, overseg=slic_map)
        return diff_superpix, sim_superpix, seg_slic, pred_R, pred_T


def clustergnet(**kwargs):
    model = ClusterNet(train_s=kwargs["train_s"])
    return model
