import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_ade20k_label_colormap():
  """Creates a label colormap used in ADE20K segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [0, 0, 0],
      [9, 7, 230],
      [0, 102, 200],
      [255, 6, 51],
      [180, 120, 120],
      [6, 230, 230],
      [4, 200, 3],
      [120, 120, 80],
      [204, 5, 255],
      [100, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [80, 50, 50],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [140, 140, 140],
      [61, 230, 250],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [120, 120, 120],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])


def visualize_rig(rig, img):
    img_draw = img.copy()
    this_level = [rig.root_id]
    while this_level:
        next_level = []
        for pid in this_level:
            cv2.circle(img_draw, (rig.joint_pos[pid][1], rig.joint_pos[pid][0]), 3, (0, 0, 255), 1)
            ch_list = np.argwhere(rig.hierarchy==pid).squeeze(axis=1)
            for chid in ch_list:
                cv2.line(img_draw, (rig.joint_pos[pid][1], rig.joint_pos[pid][0]), (rig.joint_pos[chid][1], rig.joint_pos[chid][0]), (255, 0, 0), 1)
            next_level += ch_list.tolist()
        this_level = next_level
    cv2.namedWindow("rig")
    cv2.imshow("rig", img_draw)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return img_draw


def visualize_corr(img_1, mask_1, img_2, corr_mat, show=False):
    cmap = plt.cm.get_cmap('jet')
    img1_cmap = np.ones((img_1.shape[0], img_1.shape[1], 3))
    img2_cmap = np.ones((img_2.shape[0], img_2.shape[1], 3))
    for r in np.arange(0, corr_mat.shape[0]):
        for c in np.arange(0, corr_mat.shape[1]):
            if mask_1[r, c] == 255 and corr_mat[r, c, 0] != -1:
                p_color = cmap((r * img_1.shape[0] + c) / (img_1.shape[0] * img_1.shape[1]))[0:3]
                r2 = int(corr_mat[r, c, 0])
                c2 = int(corr_mat[r, c, 1])
                img1_cmap[r, c] = np.asarray(p_color)
                img2_cmap[r2, c2] = np.asarray(p_color)
    img_cmap = np.concatenate((img1_cmap, img2_cmap), axis=1)
    img_cat = np.concatenate((img_1, img_2), axis=1)
    img_show = np.concatenate((img_cat/255.0, img_cmap), axis=0)
    if show:
        cv2.namedWindow("correspondence")
        cv2.imshow('correspondence', img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return np.round(img_show*255.0).astype(np.uint8)


def visualize_seg(img, seg_map, show=False):
    cmap = create_ade20k_label_colormap()
    seg_vis = np.ones_like(img, dtype=np.uint8)*255
    for i in range(seg_map.max()+1):
        color = cmap[i]
        pos = np.argwhere(seg_map==i)
        seg_vis[pos[:,0], pos[:,1]] = color
    if show:
        img_show = np.concatenate((img, seg_vis), axis=1)
        cv2.imshow("seg", img_show)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return seg_vis

def visualize_vismask(img1, img2, pred_vismask, gt_vismask, show=False):
    cmap = plt.cm.get_cmap('Reds')
    vis_pos_gt = np.argwhere(gt_vismask>0)
    img_vis_gt = np.ones_like(img1, dtype=np.uint8) * 255
    img_vis_gt[vis_pos_gt[:, 0], vis_pos_gt[:, 1]] = np.array([0, 0, 255])
    if pred_vismask is not None:
        img_vis_pred = np.ones_like(img1, dtype=np.uint8) * 255
        vis_pos = np.argwhere(pred_vismask > 0)
        vis_prob = pred_vismask[vis_pos[:, 0], vis_pos[:, 1]]
        vis_prob = (vis_prob - np.min(vis_prob) + 1e-8) / (np.max(vis_prob) - np.min(vis_prob)  + 1e-8)
        #vis_prob = vis_prob**4
        #vis_prob[vis_prob < 0.5] = 0.0
        img_vis_pred[vis_pos[:, 0], vis_pos[:, 1]] = np.round(cmap(vis_prob)[:, 0:3] * 255).astype(np.uint8)
        img_vis_pred = img_vis_pred[:,:,::-1]
    if pred_vismask is not None:
        img_show = np.concatenate((img1, img2, img_vis_pred, img_vis_gt), axis=1)
    else:
        img_show = np.concatenate((img1, img2, img_vis_gt), axis=1)
    if show:
        cv2.imshow("vismask", img_show)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return img_show

def visualize_cage(mask, cage, tri_face=None, show=False):
    mask_show = np.repeat(mask[..., np.newaxis], 3, axis=-1)
    for i in range(len(cage)):
        # if i == (len(cage)-1):
        #     cv2.line(mask_show, (cage[i, 1], cage[i, 0]), (cage[0, 1], cage[0, 0]), (255, 0, 0), 1)
        # else:
        #     cv2.line(mask_show, (cage[i, 1], cage[i, 0]), (cage[i+1, 1], cage[i+1, 0]), (255, 0, 0), 1)

        cv2.circle(mask_show, (cage[i, 1], cage[i, 0]), 3, (0, 0, 255), 1)
    if tri_face is not None:
        for tri in tri_face:
            v0 = cage[tri[0]]
            v1 = cage[tri[1]]
            v2 = cage[tri[2]]
            cv2.line(mask_show, (v0[1], v0[0]), (v1[1], v1[0]), (255, 0, 0), 1)
            cv2.line(mask_show, (v1[1], v1[0]), (v2[1], v2[0]), (255, 0, 0), 1)
            cv2.line(mask_show, (v2[1], v2[0]), (v0[1], v0[0]), (255, 0, 0), 1)
    if show:
        cv2.namedWindow("cage")
        cv2.imshow("cage", mask_show)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return mask_show


def visualize_warp(img_src, img_dst, img_warp):
    img_show = np.concatenate((img_src, img_dst, img_warp), axis=1)
    cv2.imshow("warp", img_show)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return img_show


def visualize_grid(pts, grid):
    img_show = np.zeros((256, 256, 3), dtype=np.uint8)
    img_show[pts[:, 0], pts[:, 1]] = np.array([255, 255, 255])
    verts = grid.v
    edges = np.argwhere(np.triu(grid.adj))
    for e in edges:
        cv2.line(img_show, (verts[e[0], 1], verts[e[0], 0]), (verts[e[1], 1], verts[e[1], 0]), (255, 0, 0), 1)
    for v in grid.v:
        cv2.circle(img_show, (v[1], v[0]), 3, (0, 0, 255), 1)
    cv2.imshow("grid", img_show)
    cv2.waitKey()
    cv2.destroyAllWindows()

