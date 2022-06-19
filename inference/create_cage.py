import cv2
import numpy as np
import torch
from scipy.spatial import ConvexHull
from utils.OBB2D import OBB2D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
