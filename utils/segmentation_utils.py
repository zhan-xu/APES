import cv2, numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import label, regionprops
from utils.vis_utils import visualize_seg


def connectivity_check(map1, map2):
    map1_dilate = cv2.dilate(map1.astype(np.uint8)*255, np.ones(3, np.uint8), iterations=1)
    map2_dilate = cv2.dilate(map2.astype(np.uint8)*255, np.ones(3, np.uint8), iterations=1)
    if (map1_dilate * map2_dilate).sum() > 0:
        return True
    else:
        return False


def get_adj_parts(seg_map, query_id, mask=None):
    return_list = []
    if mask is None:
        mask = np.ones_like(seg_map)
    part_query = (seg_map == query_id)
    for l in np.unique(seg_map[mask]):
        part_l = (seg_map == l) * mask
        if connectivity_check(part_l, part_query):
            return_list.append(l)
    return return_list



def renumber(seg_map):
    seg_map_out = -np.ones_like(seg_map)
    new_label = 0
    for l in np.unique(seg_map):
        if l == -1:
            continue
        else:
            seg_map_out[seg_map == l] = new_label
            new_label += 1
    return seg_map_out


def filter_segmentation(seg_map, slic_map, image):
    """
    1) split disconnected components from the same part, remove small parts
    2) merge isolate superpixel to one of its neighbors
    :param seg_map: H, W
    :param slic_map: H, W
    :return: H, W
    """
    # from utils.vis_utils import visualize_seg

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    total_area_thrd = 200

    # step 1: split disconnected components from the same part
    seg_map_out = seg_map.copy()
    new_label = seg_map.max() + 1
    for l in range(seg_map.max() + 1):
        mask_l = ((seg_map == l) * 255).astype(np.uint8)
        label_image = label(mask_l, connectivity=2)
        if label_image.max() > 1:
            for l in range(label_image.max() + 1):
                if ((label_image == l) & (mask_l > 0)).sum() > 0:
                    seg_map_out[label_image == l] = new_label
                    new_label += 1
    seg_map = seg_map_out

    # step 2: merge isolate superpixel to one of its neighbors
    seg_map_out = seg_map.copy()
    missing_label = -2
    for l in range(seg_map.max() + 1):
        mask_l = (seg_map == l)
        if len(np.unique(slic_map[mask_l])) == 1:
            seg_map_out[mask_l] = missing_label
            missing_label -= 1
    if seg_map_out.min() < -1:
        for l in np.arange(seg_map_out.min(), -1):
            neighbor_IDs = get_adj_parts(seg_map_out, l, seg_map_out > -1)
            if len(neighbor_IDs) == 0:
                seg_map_out[seg_map_out == l] = seg_map[seg_map_out == l]
                continue
            if len(neighbor_IDs) == 1:
                nearest_ID = neighbor_IDs[0]
            else:
                color_features = []
                for neighbor_id in neighbor_IDs:
                    hist = cv2.calcHist([image_gray], [0], (seg_map_out == neighbor_id).astype(np.uint8) * 255, [50], [0, 256])
                    hist = hist / hist.sum()
                    color_features.append(hist)
                color_features = np.concatenate(color_features, axis=-1)
                hist = cv2.calcHist([image_gray], [0], (seg_map_out == l).astype(np.uint8) * 255, [50], [0, 256])
                color_feature_query = hist / hist.sum()
                nearest_ID = neighbor_IDs[np.argmin(((color_features - color_feature_query) ** 2).sum(0))]
            seg_map_out[seg_map_out == l] = nearest_ID
    seg_map = renumber(seg_map_out)

    # step 3: remove small parts
    seg_map_out = seg_map.copy()
    for l in range(seg_map.max() + 1):
        mask_l = ((seg_map == l) * 255).astype(np.uint8)
        if np.sum(mask_l > 0) < total_area_thrd:
            seg_map_out[mask_l] = -1

    # img_seg = visualize_seg(image, seg_map_out, show=False)
    # img_slic = mark_boundaries(img_seg, slic_map)
    # cv2.imshow("img", np.concatenate((image, img_seg, np.round(img_slic * 255).astype(np.uint8)), axis=1))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return seg_map_out


def sample_from_slic(slic_map, group_size):
    src_pixel_group = []
    for label in np.arange(slic_map.max() + 1):
        src_pix_pos = np.argwhere(slic_map == label)
        if len(src_pix_pos) >= group_size:
            randn_idx = np.random.choice(np.arange(len(src_pix_pos)), group_size, replace=False)
            src_pix_pos_sample = src_pix_pos[randn_idx]
        else:
            randn_idx = np.random.choice(np.arange(len(src_pix_pos)), group_size - len(src_pix_pos), replace=True)
            src_pix_pos_sample = np.concatenate((src_pix_pos, src_pix_pos[randn_idx]), axis=0)
        src_pixel_group.append(src_pix_pos_sample)
    src_pixel_group = np.stack(src_pixel_group, axis=0)
    src_pixel_group = src_pixel_group / np.array([[[slic_map.shape[0], slic_map.shape[1]]]])
    return src_pixel_group
