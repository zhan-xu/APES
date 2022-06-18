import sys
sys.path.append("./")
import torch
import models
import os, numpy as np
from tqdm import tqdm
from datasets.okay_samurai_pair import OkaySamuraiPair
from torch.utils.data import DataLoader
import torch_cluster.knn as knn
from torch_scatter import scatter_add
from utils.vis_utils import visualize_corr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def output_pred_corr(data_loader, model, outdir):
    for data_pack in tqdm(data_loader):
        img_1, img_2 = data_pack["img1"].to(device), data_pack["img2"].to(device)
        mask_1, mask_2 = data_pack["mask1"].float().to(device), data_pack["mask2"].float().to(device)
        img_names = data_pack["img1_filename"]
        with torch.no_grad():
            img1_feature, img2_feature, tau = model(img_1, img_2, mask_1.unsqueeze(1), mask_2.unsqueeze(1))

            masked_pos_src = torch.nonzero(mask_1)
            masked_pos_dst = torch.nonzero(mask_2)
            feat_x = img1_feature[masked_pos_src[:, 0], :, masked_pos_src[:, 1], masked_pos_src[:, 2]]
            feat_y = img2_feature[masked_pos_dst[:, 0], :, masked_pos_dst[:, 1], masked_pos_dst[:, 2]]
            batch_x = masked_pos_src[:, 0]
            batch_y = masked_pos_dst[:, 0]

            assign_xy = knn(feat_y, feat_x, 3, batch_y, batch_x, cosine=True)
            # sim_xy = torch.sum(feat_x[assign_xy[0]] * feat_y[assign_xy[1]], dim=-1, keepdim=True) / tau
            # sim_xy = torch.exp(sim_xy - sim_xy.max())
            # pt_forward_match = scatter_add(masked_pos_dst[assign_xy[1]] * sim_xy, assign_xy[0], dim=0) / scatter_add(sim_xy, assign_xy[0], dim=0)

            # In practice, we find the following way is better. sim_xy is always postive since we only consider top-3 most similar pairs.
            sim_xy = torch.sum(feat_x[assign_xy[0]] * feat_y[assign_xy[1]], dim=-1, keepdim=True)
            pt_forward_match = scatter_add(masked_pos_dst[assign_xy[1]] * sim_xy, assign_xy[0], dim=0) / scatter_add(sim_xy, assign_xy[0], dim=0)

            assign_yx = knn(feat_x, feat_y, 3, batch_x, batch_y, cosine=True)
            # sim_yx = torch.sum(feat_y[assign_yx[0]] * feat_x[assign_yx[1]], dim=-1, keepdim=True) / tau
            # sim_yx = torch.exp(sim_yx - sim_yx.max())
            # pt_backward_match = scatter_add(masked_pos_src[assign_yx[1]] * sim_yx, assign_yx[0], dim=0) / scatter_add(sim_yx, assign_yx[0], dim=0)

            sim_yx = torch.sum(feat_y[assign_yx[0]] * feat_x[assign_yx[1]], dim=-1, keepdim=True)
            pt_backward_match = scatter_add(masked_pos_src[assign_yx[1]] * sim_yx, assign_yx[0], dim=0) / scatter_add(sim_yx, assign_yx[0], dim=0)

        corr_forward_pred = -np.ones((mask_1.shape[0], mask_1.shape[1], mask_1.shape[2], 2)).astype(np.int64)
        corr_forward_pred[masked_pos_src[:, 0].cpu().numpy(), masked_pos_src[:, 1].cpu().numpy(), masked_pos_src[:, 2].cpu().numpy()] = \
            pt_forward_match[:, 1:].cpu().numpy()
        corr_backward_pred = -np.ones((mask_2.shape[0], mask_2.shape[1], mask_2.shape[2], 2)).astype(np.int64)
        corr_backward_pred[masked_pos_dst[:, 0].cpu().numpy(), masked_pos_dst[:, 1].cpu().numpy(), masked_pos_dst[:, 2].cpu().numpy()] = \
            pt_backward_match[:, 1:].cpu().numpy()

        for i in range(len(img_1)):
            img_name = img_names[i]
            model_id = img_name.split(".")[0]
            #if os.path.exists(os.path.join(outdir, f"{model_id}_predcorr.npy")):
            #    continue
            #print("processing: ", img_name)
            img_1_np = img_1[i].to("cpu").numpy().transpose(1, 2, 0)
            img_2_np = img_2[i].to("cpu").numpy().transpose(1, 2, 0)
            img_1_np = np.round(img_1_np * 255.0).astype(np.uint8)
            img_2_np = np.round(img_2_np * 255.0).astype(np.uint8)
            mask_1_np = (data_pack["mask1"][i] * 255).to("cpu").numpy()
            mask_2_np = (data_pack["mask2"][i] * 255).to("cpu").numpy()

            # forward
            corr_pred_forward_i = corr_forward_pred[i]
            #img = visualize_corr(img_1_np, mask_1_np, img_2_np, corr_pred_forward_i, show=True)

            # backward
            corr_pred_backward_i = corr_backward_pred[i]
            #img = visualize_corr(img_2_np, mask_2_np, img_1_np, corr_pred_backward_i, show=True)

            np.save(os.path.join(outdir, f"{model_id}_predcorr.npy"), np.concatenate((corr_pred_forward_i, corr_pred_backward_i), axis=-1))


if __name__ == "__main__":
    for split_name in ["train", "val", "test"]:
        data_folder = f"/mnt/DATA_LINUX/zhan/puppets/okay_samurai/{split_name}/"
        model = models.__dict__["corrnet"](n_channels=4, n_classes=64, bilinear=True)
        checkpoint = torch.load("checkpoints/pretrain_corrnet_os/model_best.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        model.eval()  # switch to test mode
        model.to(device)

        data_loader = DataLoader(OkaySamuraiPair(root=data_folder, train_corr=True), batch_size=4, shuffle=False)
        output_pred_corr(data_loader, model, data_folder)