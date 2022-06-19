import sys
sys.path.append("./")
import argparse, os, cv2, shutil, numpy as np, time, glob
from tqdm import tqdm
from itertools import permutations, combinations, groupby
import torch, torch.nn.functional as F, torch.backends.cudnn as cudnn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from utils.os_utils import isdir, mkdir_p, isfile
import models
from models.clusternet import sync_motion_seg
from inference.part_select import PartSelector
from training.train_fullnet import make_coordinate_grid, slic_seg_to_pixel, save_checkpoint
from utils.vis_utils import visualize_seg, visualize_corr
from skimage import segmentation
from skimage.measure import label, regionprops
from utils.segmentation_utils import filter_segmentation, sample_from_slic
from scipy.io import savemat, loadmat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def remove_noise(mask_in):
    mask_out = mask_in.copy()
    label_image = label(mask_in, connectivity=2)
    if label_image.max() > 1:
        for l in range(label_image.max() + 1):
            if (label_image == l).sum() < 100:
                mask_out[label_image == l] = 0
    return mask_out


class TestData(Dataset):
    def __init__(self, root):
        self.num_frame = len(glob.glob(os.path.join(root, "*_mask.png")))
        self.pairs = list(combinations(np.arange(self.num_frame), 2))
        self.img_list, self.mask_list, self.slic_list, self.src_pixel_group_all = [], [], [], []
        for i in range(self.num_frame):
            char_name = root.split("/")[-1]
            img_i = cv2.imread(os.path.join(root, f"{char_name}_{i}.png"))
            mask_i = cv2.imread(os.path.join(root, f"{char_name}_{i}_mask.png"), 0)
            mask_i = remove_noise(mask_i)
            mask_i = (mask_i > 50)
            slic_i = segmentation.slic(img_i, n_segments=50, mask=mask_i, start_label=1, min_size_factor=0.5,
                                       enforce_connectivity=False, compactness=35, max_iter=200, sigma=0)
            slic_i -= 1
            # img_slic = segmentation.mark_boundaries(img_i, slic_i)
            # cv2.imshow("img", img_slic)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            src_pixel_group_i = sample_from_slic(slic_i, group_size=120)
            self.img_list.append(img_i)
            self.mask_list.append(mask_i)
            self.slic_list.append(slic_i)
            self.src_pixel_group_all.append(src_pixel_group_i)
        self.img1_all, self.img2_all, self.mask1_all, self.mask2_all, \
        self.slic1_all, self.slic2_all, self.src_pixel_group_1_all, self.src_pixel_group_2_all = ([] for i in range(8))
        for pair in self.pairs:
            self.img1_all.append(self.img_list[pair[0]].copy())
            self.img2_all.append(self.img_list[pair[1]].copy())
            self.mask1_all.append(self.mask_list[pair[0]].copy())
            self.mask2_all.append(self.mask_list[pair[1]].copy())
            self.slic1_all.append(self.slic_list[pair[0]].copy())
            self.slic2_all.append(self.slic_list[pair[1]].copy())
            self.src_pixel_group_1_all.append(self.src_pixel_group_all[pair[0]].copy())
            self.src_pixel_group_2_all.append(self.src_pixel_group_all[pair[1]].copy())
        self.img1_all = np.stack(self.img1_all, axis=0)
        self.img2_all = np.stack(self.img2_all, axis=0)
        self.mask1_all = np.stack(self.mask1_all, axis=0)
        self.mask2_all = np.stack(self.mask2_all, axis=0)
        self.slic1_all = np.stack(self.slic1_all, axis=0)
        self.slic2_all = np.stack(self.slic2_all, axis=0)
        self.src_pixel_group_1_all = np.stack(self.src_pixel_group_1_all, axis=0)
        self.src_pixel_group_2_all = np.stack(self.src_pixel_group_2_all, axis=0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1 = T.ToTensor()(self.img1_all[idx])
        img2 = T.ToTensor()(self.img2_all[idx])
        mask1 = self.mask1_all[idx]
        mask2 = self.mask2_all[idx]
        slic1 = self.slic1_all[idx]
        slic2 = self.slic2_all[idx]
        src_pixel_group1 = self.src_pixel_group_1_all[idx].astype(np.float32)
        src_pixel_group2 = self.src_pixel_group_2_all[idx].astype(np.float32)
        pair = self.pairs[idx]
        return {"img1": img1, "img2": img2, "mask1": mask1, "mask2": mask2, "slic1": slic1, "slic2": slic2,
                "src_pixel_group1": src_pixel_group1, "src_pixel_group2": src_pixel_group2, "pair": pair}


def affinity2seg(A_in, slic_in, img1_np, mask):
    pred_G_i = sync_motion_seg(torch.sigmoid(A_in).unsqueeze(0), cut_thres=0.01).squeeze(0)
    pred_G_i = pred_G_i / (pred_G_i.sum(-1, keepdim=True) + 1e-6)
    seg_pred_f = slic_seg_to_pixel(slic_in, pred_G_i)  # H, W, S
    seg_pred_f = torch.argmax(seg_pred_f, -1)
    seg_pred_f[mask == False] = -1
    seg_pred_np = seg_pred_f.to("cpu").numpy()
    try:
        seg_pred_np = filter_segmentation(seg_pred_np, slic_in.to("cpu").numpy(), img1_np)
    except:
        return None
    if seg_pred_np.max() == 0:
        return None
    return seg_pred_np


def select(model, test_dataset, test_loader, output_folder=None):
    parts_database = []
    pred_corr_all = {}
    for data_pack in tqdm(test_loader):
        img_1, img_2 = data_pack["img1"].to(device), data_pack["img2"].to(device)
        mask_1, mask_2 = data_pack["mask1"].to(device), data_pack["mask2"].to(device)
        slic_1, slic_2 = data_pack["slic1"].to(device), data_pack["slic2"].to(device)
        src_pixel_group_1, src_pixel_group_2 = data_pack["src_pixel_group1"].to(device), data_pack["src_pixel_group2"].to(device)
        pair_name = data_pack["pair"]
        with torch.no_grad():
            res = model.test_run(img_1, img_2, mask_1, mask_2, slic_1, slic_2, src_pixel_group_1, src_pixel_group_2,
                                 pair_name=pair_name, debug_folder=output_folder)
        for i in range(len(img_1)):
            img_1_np = np.round(img_1[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_2_np = np.round(img_2[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # forward
            pair = (pair_name[0][i].item(), pair_name[1][i].item())
            pred_corr_all[f"{pair[0]}{pair[1]}"] = res["pred_corr"][i, ..., :2].to("cpu").numpy()
            seg_pred_f_np = affinity2seg(res["A_f"][i], res["slic_f"][i], img_1_np, mask_1[i])
            if seg_pred_f_np is not None:
                if False:  # output intermediate for debug
                    slic_np = res["slic_f"][i].to("cpu").numpy()
                    img_seg = visualize_seg(img_1_np, seg_pred_f_np, show=False)
                    img_slic = segmentation.mark_boundaries(img_1_np, slic_np)
                    cv2.imwrite(os.path.join(output_folder, f"seg_{pair[0]}{pair[1]}.png"),
                                np.concatenate((np.round(img_slic * 255).astype(np.uint8), img_seg), axis=1))
                    np.save(os.path.join(output_folder, f"seg_{pair[0]}{pair[1]}.npy"), seg_pred_f_np)
                parts_database.append({"img_id": pair[0], "seg": seg_pred_f_np})
            else:
                print("Invalid Segmentation. Skipping ->", pair)

            # backward
            pair = (pair_name[1][i].item(), pair_name[0][i].item())
            pred_corr_all[f"{pair[0]}{pair[1]}"] = res["pred_corr"][i, ..., 2:].to("cpu").numpy()
            seg_pred_b_np = affinity2seg(res["A_b"][i], res["slic_b"][i], img_2_np, mask_2[i])
            if seg_pred_b_np is not None:
                if False:  # output intermediate for debug
                    # debug
                    slic_np = res["slic_b"][i].to("cpu").numpy()
                    img_seg = visualize_seg(img_2_np, seg_pred_b_np, show=False)
                    img_slic = segmentation.mark_boundaries(img_2_np, slic_np)
                    cv2.imwrite(os.path.join(output_folder, f"seg_{pair[0]}{pair[1]}.png"),
                                np.concatenate((np.round(img_slic * 255).astype(np.uint8), img_seg), axis=1))
                    np.save(os.path.join(output_folder, f"seg_{pair[0]}{pair[1]}.npy"), seg_pred_b_np)
                parts_database.append({"img_id": pair[0], "seg": seg_pred_b_np})
            else:
                print("Invalid Segmentation. Skipping ->", pair)

    selector = PartSelector(np.stack(test_dataset.img_list, 0), np.stack(test_dataset.mask_list, 0),
                            parts_database, pred_corr_all, num_trial=100)

    selector.part_database = selector.create_database()
    paths = selector.select(test_dataset.slic_list, filter_thrd=0.45, debug_folder=output_folder)
    for i in range(len(paths)):
        mkdir_p(os.path.join(output_folder, f"selection_{i}/"))
        for step in paths[i]:
            part_step = selector.deformed_parts[step]
            mask_step = selector.deformed_masks[step]
            for img_id in range(part_step.shape[0]):
                recons_eval_img = np.where(np.repeat(mask_step[img_id, :, :, np.newaxis], 3, axis=-1), part_step[img_id],
                                           np.ones_like(part_step[img_id], dtype=np.uint8) * 255)
                cv2.imwrite(os.path.join(output_folder, f"selection_{i}/part_{step}_img_{img_id}.png"), recons_eval_img)
                cv2.imwrite(os.path.join(output_folder, f"selection_{i}/part_{step}_img_{img_id}_mask.png"), selector.deformed_masks[step][img_id])

    picked_imgs, picked_masks, parts_img_warped, parts_mask_warped = selector.reconstruct(debug_folder=output_folder)
    mkdir_p(os.path.join(output_folder, "selection_by_deform/"))
    for i in range(len(picked_masks)):
        cv2.imwrite(os.path.join(output_folder, f"selection_by_deform/part_{i}_img.png"), picked_imgs[i])
        cv2.imwrite(os.path.join(output_folder, f"selection_by_deform/part_{i}_mask.png"), picked_masks[i])
    np.save(os.path.join(output_folder, f"selection_by_deform/parts_img_warped.npy"), parts_img_warped)
    np.save(os.path.join(output_folder, f"selection_by_deform/parts_mask_warped.npy"), parts_mask_warped)


def main(args):
    # initialize model
    model = models.__dict__[args.arch](n_channels=4, n_classes=64, bilinear=True, train_s=args.train_s, offline_corr=False)
    model.to(device)

    if isfile(args.init_weight_path):
        checkpoint = torch.load(args.init_weight_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.init_weight_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.init_weight_path))
        exit()

    cudnn.benchmark = True
    model.eval()
    char_name_list = os.listdir(args.test_folder)
    for char_name in char_name_list:
        print(char_name)

        # prepare data
        test_dataset = TestData(root=os.path.join(args.test_folder, f"{char_name}"))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # select parts
        output_folder = os.path.join(args.output_folder, f"{char_name}/")
        mkdir_p(output_folder)
        select(model, test_dataset, test_loader, output_folder=output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='fullnet')
    parser.add_argument('--init_weight_path', default="../checkpoints/train_fullnet/model_best.pth.tar", type=str)
    parser.add_argument('--test_folder', default='/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/okaysamural_sheets/',
                        type=str, help='folder of testing data')
    parser.add_argument('--output_folder', type=str, help='output folder',
                        default='/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/results/train_cluster_os/')
    parser.add_argument('--train_s', default=12, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    print(parser.parse_args())
    main(parser.parse_args())
