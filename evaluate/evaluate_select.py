import os, glob, numpy as np, cv2
import torch
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from utils.vis_utils import visualize_seg
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import lpips

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hungarian_matching(pred_seg, gt_seg):
    intersect = (np.expand_dims(pred_seg, -1) * np.expand_dims(gt_seg, -2)).sum(axis=0).sum(axis=0)
    matching_cost = 1-np.divide(intersect, pred_seg.sum(0).sum(0)[:, None]+gt_seg.sum(0).sum(0)[None, :]-intersect+1e-8)
    row_ind, col_ind = linear_sum_assignment(matching_cost)
    return np.vstack((row_ind, col_ind))


def load_gt(char_data_folder, char_name):
    img_filelist = glob.glob(os.path.join(char_data_folder, f"*_mask.png"))
    img_array = []
    seg_array = []
    for i in range(len(img_filelist)):
        img_array.append(cv2.imread(os.path.join(char_data_folder, f"{char_name}_{i}.png")))
        seg_array.append(np.load(os.path.join(char_data_folder, f"{char_name}_{i}_seg.npy")))
    return np.stack(img_array, axis=0), np.stack(seg_array, axis=0)


def cal_iou(rec_segs, tar_segs):
    iou = []
    for i in range(len(rec_segs)):
        rec_seg_i_onehot = np.zeros((rec_segs.shape[1], rec_segs.shape[2], rec_segs[i].max()+1), dtype=np.uint8)
        tar_seg_i_onehot = np.zeros((tar_segs.shape[1], tar_segs.shape[2], tar_segs[i].max()+1), dtype=np.uint8)
        for col_id, label in enumerate(range(rec_segs[i].max()+1)):
            label_pos = np.argwhere(rec_segs[i]==label)
            rec_seg_i_onehot[label_pos[:, 0], label_pos[:, 1], col_id] = 1
        for col_id, label in enumerate(range(tar_segs[i].max()+1)):
            label_pos = np.argwhere(tar_segs[i]==label)
            tar_seg_i_onehot[label_pos[:, 0], label_pos[:, 1], col_id] = 1
        matching_id_i = hungarian_matching(rec_seg_i_onehot, tar_seg_i_onehot)
        pred_seg_i_reorder = rec_seg_i_onehot[..., matching_id_i[0]]
        gt_seg_i_reorder = tar_seg_i_onehot[..., matching_id_i[1]]
        interset = (pred_seg_i_reorder * gt_seg_i_reorder).sum(0).sum(0)
        iou_i = np.divide(interset, pred_seg_i_reorder.sum(0).sum(0) + gt_seg_i_reorder.sum(0).sum(0) - interset + 1e-8)
        iou.append(iou_i.mean())
    #print("char_id with maximal iou:", np.argmax(np.array(iou)))
    return np.array(iou).mean()


def cal_perceptual_loss(rec_imgs, tar_imgs, loss_fn):
    rec_imgs_tensor = torch.from_numpy(2 * (rec_imgs / 255.0) - 1).permute(0, 3, 1, 2).to(device)
    tar_imgs_tensor = torch.from_numpy(2 * (tar_imgs / 255.0) - 1).permute(0, 3, 1, 2).to(device)
    d = loss_fn(rec_imgs_tensor.float(), tar_imgs_tensor.float()).squeeze()
    return d.mean().item()

def calc_pnsr(rec_imgs, tar_imgs):
    pnsr_loss = []
    for i in range(len(rec_imgs)):
        d = peak_signal_noise_ratio(tar_imgs[i], rec_imgs[i], data_range=255)
        pnsr_loss.append(d)
    return np.array(pnsr_loss).mean()

def evaluate(res_folder, data_folder):
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    char_list = next(os.walk(res_folder))[1]
    mse_all, per_all, iou_all, pnsr_all, cov_all = [], [], [], [], []
    for char_name in char_list:
        print(char_name)
        char_data_folder = os.path.join(data_folder, f"{char_name}")
        char_res_folder = os.path.join(res_folder, f"{char_name}")
        tar_imgs, tar_segs = load_gt(char_data_folder, char_name)
        parts_imgs = np.load(os.path.join(char_res_folder, "selection_by_deform/parts_img_warped.npy"))
        parts_imgs = np.round(parts_imgs * 255).astype(np.uint8)
        parts_masks = np.load(os.path.join(char_res_folder, "selection_by_deform/parts_mask_warped.npy"))
        rec_imgs = np.ones_like(tar_imgs) * 255
        rec_segs = -np.ones_like(tar_segs)
        for p in range(parts_imgs.shape[1]):
            rec_imgs = np.where(parts_masks[:, p, ...] > 0.2, parts_imgs[:, p, ...], rec_imgs)
            rec_segs = np.where(np.mean(parts_masks[:, p, ...], axis=-1) > 0.2, p, rec_segs)

        # cv2.imshow("rec", np.concatenate((np.concatenate(list(rec_imgs), axis=1), np.concatenate(list(tar_imgs), axis=1)), axis=0))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        char_per = cal_perceptual_loss(rec_imgs, tar_imgs, loss_fn_vgg)
        char_mse = ((rec_imgs.astype(np.float32) - tar_imgs.astype(np.float32))**2).mean()
        char_iou = cal_iou(rec_segs, tar_segs)
        char_pnsr = calc_pnsr(rec_imgs, tar_imgs)
        mse_all.append(char_mse)
        iou_all.append(char_iou)
        per_all.append(char_per)
        pnsr_all.append(char_pnsr)
    print("MSE:", np.array(mse_all).mean(), "IOU", np.array(iou_all).mean(), "PNSR", np.array(pnsr_all).mean(),
          "PERCEPTUAL:", np.array(per_all).mean())


if __name__ == "__main__":
    res_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/results/train_cluster_os/"
    data_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test_sheet/"
    evaluate(res_folder, data_folder)
