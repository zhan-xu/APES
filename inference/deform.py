import cv2
import numpy as np
import torch
from utils.vis_utils import visualize_corr, visualize_cage
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from utils.OBB2D import OBB2D
import models
from scipy.spatial import Delaunay
from inference.diff_renderer import py3d_warp_image
from inference.ARAP import ARAP_Deformer as CageDeformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def visualize_corr_one_seg(img_1, mask_1, seg_1, img_2, corr_pred, label):
    cmap = plt.cm.get_cmap('jet')
    img1_cmap = np.ones((img_1.shape[0], img_1.shape[1], 3))
    img2_cmap = np.ones((img_2.shape[0], img_2.shape[1], 3))
    for r in np.arange(0, corr_pred.shape[0]):
        for c in np.arange(0, corr_pred.shape[1]):
            if mask_1[r, c] == 255 and seg_1[r, c] == label and corr_pred[r, c, 2] >= 0:
                p_color = cmap((r * img_1.shape[0] + c) / (img_1.shape[0] * img_1.shape[1]))[0:3]
                r2 = int(corr_pred[r, c, 0])
                c2 = int(corr_pred[r, c, 1])
                img1_cmap[r, c] = np.asarray(p_color)
                img2_cmap[r2, c2] = np.asarray(p_color)

    img_cmap = np.concatenate((img1_cmap, img2_cmap), axis=1)
    # cv2.imshow(f'correspondence', img_cmap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return np.round(img_cmap*255.0).astype(np.uint8)


def create_cage(img, mask=None, seg=None, part_label=None, type="hull", interval=None):
    if mask is None and seg is not None and part_label is not None:
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        mask[seg==part_label] = 255
    elif mask is not None:
        pass
    else:
        raise NotImplementedError
    mask_expand = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=5)
    pts = np.argwhere(mask_expand==255)
    if type == "hull":
        hull = ConvexHull(pts)
        anchors = pts[hull.vertices]
        dist = np.sqrt(np.sum((anchors[np.newaxis, ...] - anchors[:, np.newaxis, :])**2, axis=-1))
        dist += 1000*np.eye(len(dist))
        while np.min(dist) < 10:
            #print(np.min(dist))
            merge_ids = np.argwhere(dist == np.min(dist))[0]
            anchors_new = []
            for r in range(len(anchors)):
                if r not in merge_ids:
                    anchors_new.append(anchors[r])
                elif r == merge_ids[0]:
                    anchors_new.append(np.round((anchors[merge_ids[0]]+anchors[merge_ids[1]]) / 2).astype(int))
                else:
                    continue
            anchors = np.array(anchors_new)
            dist = np.sqrt(np.sum((anchors[np.newaxis, ...] - anchors[:, np.newaxis, :]) ** 2, axis=-1))
            dist += 1000 * np.eye(len(dist))
    elif type == "aabb":
        anchors = np.array([[np.min(pts[:, 0]), np.min(pts[:, 1])],
                            [np.min(pts[:, 0]), np.max(pts[:, 1])],
                            [np.max(pts[:, 0]), np.max(pts[:, 1])],
                            [np.max(pts[:, 0]), np.min(pts[:, 1])]])
    elif type == "obb_grid":
        obb = OBB2D(pts)
        if interval is None:
            anchors = np.round(obb.grids()).astype(np.int64)
        else:
            anchors = np.round(obb.grids(interval)).astype(np.int64)
    else:
        raise NotImplementedError
    return mask, anchors


def infer_corr_rgb(model, color1, color2, mask1, mask2):
    color1 = color1 / 255.0
    color2 = color2 / 255.0
    color1 = np.transpose(color1, (2, 0, 1)).astype(np.float32)
    color2 = np.transpose(color2, (2, 0, 1)).astype(np.float32)
    color1 = torch.from_numpy(color1[np.newaxis,...]).to(device)
    color2 = torch.from_numpy(color2[np.newaxis,...]).to(device)
    with torch.no_grad():
        feature1, feature2, tau = model(color1, color2)
    feature1 = feature1.to("cpu").numpy().squeeze(axis=0).transpose(1, 2, 0)
    feature2 = feature2.to("cpu").numpy().squeeze(axis=0).transpose(1, 2, 0)
    corr_pred = np.zeros((feature1.shape[0], feature1.shape[1], 3), dtype=np.int64)
    p_bound1 = np.argwhere(mask1==255)
    p_bound2 = np.argwhere(mask2==255)
    pf_from = feature1[p_bound1[:, 0], p_bound1[:, 1]]
    pf_to = feature2[p_bound2[:, 0], p_bound2[:, 1]]
    similarity = np.matmul(pf_from, pf_to.T) / tau.item()
    pairwise_nnind = np.argmax(similarity, axis=1)
    corr_pred[p_bound1[:, 0], p_bound1[:, 1], 0:2] = p_bound2[pairwise_nnind]
    corr_pred[p_bound1[:, 0], p_bound1[:, 1], 2] = 1
    return corr_pred


def resize_corr(corr_in, new_shape):
    corr_new = -np.ones((new_shape[0], new_shape[1], 3), dtype=np.int64)
    pts_from = np.argwhere(corr_in[:, :, 0]  >= 0)
    pts_to = corr_in[pts_from[:, 0], pts_from[:, 1], 0:2]
    pts_to_seg = corr_in[pts_from[:, 0], pts_from[:, 1], 2]
    pts_from = pts_from * (256.0 / 400)
    pts_to = pts_to * (256.0 / 400)
    pts_from = np.clip(np.round(pts_from), 0, new_shape[0]-1).astype(np.int64)
    pts_to = np.clip(np.round(pts_to), 0, new_shape[0]-1).astype(np.int64)
    corr_new[pts_from[:, 0], pts_from[:, 1], 0:2] = pts_to
    corr_new[pts_from[:, 0], pts_from[:, 1], 2] = pts_to_seg
    return corr_new


def boundary_chamfer_dist(mask_pred, mask_gt):
    mask_pred = cv2.medianBlur(mask_pred, 3)
    mask_gt = cv2.medianBlur(mask_gt, 3)
    mask_pred_bound = mask_pred - cv2.erode(mask_pred, np.ones((3, 3), np.uint8))
    mask_gt_bound = mask_gt - cv2.erode(mask_gt, np.ones((3, 3), np.uint8))
    pts_pred_bound = np.argwhere(mask_pred_bound) / mask_pred.shape[0]
    pts_gt_bound = np.argwhere(mask_gt_bound) / mask_gt.shape[0]
    distmap = np.sqrt(np.sum((pts_pred_bound[:, np.newaxis,:] - pts_gt_bound[np.newaxis, ...])**2, axis=-1))
    chamfer_dist = (np.min(distmap, axis=0).mean() + np.min(distmap, axis=1).mean()) / 2.0
    return chamfer_dist


def deform(color_1, mask_1, seg_1, color_2, corr, visualization=False, return_chamfer=False, w_reg=0.0001):
    """
    Deform color 1 to color_2 part by part
    :param color_1: src igridsmage (H,W,3) in [0, 255]
    :param mask_1: binary src mask (H, W)
    :param seg_1: src segmentation map (H, W) in [0, num_parts]
    :param color_2: dst image  (H,W,3) in [0, 255]
    :param corr: correspondence map from color_1 to color_2. (H,W,3). The first two channels indicate the corresponding
    coordinates in the color_2. The last channel indicates visible probability in [0, 1]
    :param visualization: whether show the visualization
    :return: deformed color_1 as close as possible to color_2
    """
    img_deform = np.ones((corr.shape[0], corr.shape[0], 3), dtype=np.uint8) * 255
    if return_chamfer:
        chamfer_map = np.zeros((corr.shape[0], corr.shape[0]))
    else:
        chamfer_map = None
    for label in np.arange(seg_1.max() + 1):
        mask_part_1, cage_1 = create_cage(color_1, seg=seg_1, part_label=label, type="obb_grid")  # obb_grid, hull

        if visualization:
            tri_vis = Delaunay(cage_1 / 255.0)
            img_cage_src = visualize_cage(mask_part_1, cage_1, tri_face=tri_vis.simplices)
            img_corr = visualize_corr_one_seg(color_1, mask_1, seg_1, color_2, corr, label)

        pts_src = np.argwhere(np.logical_and(seg_1 == label, corr[:, :, -1] > 0))
        pts_tar = corr[pts_src[:, 0], pts_src[:, 1], 0:2]

        if len(pts_src) == 0:
            continue

        pts_src_float = pts_src / corr.shape[0]
        pts_tar_float = pts_tar / corr.shape[0]

        cage_deformer = CageDeformer(src_pts=torch.from_numpy(pts_src_float).float(),
                                     tar_pts=torch.from_numpy(pts_tar_float).float(),
                                     src_anchor=torch.from_numpy(cage_1 / corr.shape[0]).float())
        cage_2, pts_deform = cage_deformer.run(num_iter=300, lr=5e-2, w_reg=w_reg)  # w_reg=0.0001 for edge_lap, w_reg=0.5 for neighbor_lap
        cage_2 = cage_2.detach().numpy()
        cage_2 = np.round(cage_2 * corr.shape[0]).astype(np.int64)
        pts_deform = pts_deform.detach().numpy()
        pts_deform = np.round(pts_deform * corr.shape[0]).astype(np.int64)

        # warp
        img_warp = py3d_warp_image(color_1, cage_1, cage_2, visualize=False)
        mask_part_2 = py3d_warp_image(255-mask_part_1, cage_1, cage_2, visualize=False)
        mask_part_2 = np.any(255-mask_part_2, axis=-1).astype(np.uint8) * 255
        img_deform = np.where(np.repeat(mask_part_2[:, :, np.newaxis], 3, axis=-1), img_warp, img_deform)

        if return_chamfer:
            chamfer_i = boundary_chamfer_dist(mask_part_2, mask_part_1)
            chamfer_map += chamfer_i

        if visualization:
            img_cage_dst = visualize_cage(mask_part_2, cage_2, tri_face=tri_vis.simplices)
            masked_pixel = np.argwhere(mask_part_1)
            img_part_1 = np.ones_like(color_1, dtype=np.uint8) * 255
            img_part_1[masked_pixel[:, 0], masked_pixel[:, 1]] = color_1[masked_pixel[:, 0], masked_pixel[:, 1]]
            img_show = np.concatenate((img_corr, img_cage_src, img_cage_dst, img_part_1, img_deform), axis=1)
            cv2.namedWindow("img_deform")
            cv2.imshow("img_deform", img_show)
            cv2.waitKey()
            cv2.destroyAllWindows()
            #cv2.imwrite(f"{label}_deform.png", img_show)

    return img_deform, chamfer_map


def main():
    # load data and resize
    model_id = "hackbot_0"
    color_1 = cv2.imread(f"/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test/{model_id}_img_1.png")
    color_2 = cv2.imread(f"/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test/{model_id}_img_2.png")
    mask_1 = cv2.imread(f"/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test/{model_id}_mask_1.png", 0)
    mask_2 = cv2.imread(f"/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test/{model_id}_mask_2.png", 0)
    seg_1 = np.load(f"/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test/{model_id}_seg_1.npy")
    seg_2 = np.load(f"/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test/{model_id}_seg_2.npy")
    corr_gt = np.load(f"/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test/{model_id}_corr.npy")
    color_1 = cv2.resize(color_1, (256, 256))
    color_2 = cv2.resize(color_2, (256, 256))
    mask_1 = cv2.resize(mask_1, (256, 256))
    mask_2 = cv2.resize(mask_2, (256, 256))
    _, mask_1 = cv2.threshold(mask_1, 127, 255, cv2.THRESH_BINARY)
    _, mask_2 = cv2.threshold(mask_2, 127, 255, cv2.THRESH_BINARY)
    seg_1 = cv2.resize(seg_1, (256, 256), 0, 0, interpolation=cv2.INTER_NEAREST)
    seg_2 = cv2.resize(seg_2, (256, 256), 0, 0, interpolation=cv2.INTER_NEAREST)
    corr_gt = resize_corr(corr_gt, (256, 256))

    # run inference
    model = models.__dict__["unet1"](n_channels=3, n_classes=32, bilinear=True)
    model.to(device)
    checkpoint = torch.load("../checkpoints/unet1/model_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    corr_pred_backward = infer_corr_rgb(model, color_2, color_1, mask_2, mask_1)
    #img = visualize_corr(color_2, mask_2, color_1, corr_pred_backward)

    img_deform, _ = deform(color_2, mask_2, seg_2, color_1, corr_pred_backward, visualization=True)
    cv2.imwrite(f"{model_id}_deform.png", img_deform)


if __name__ == "__main__":
    main()
