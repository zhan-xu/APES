import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as T


class CreativeFlowPair(Dataset):
    def __init__(self, root, color_jittering=False):
        self.color_jittering = color_jittering
        self.img0_list = glob.glob(os.path.join(root, "*_0.png"))
        if len(root.split('/')[-1]) > 0:
            print("loading:", root.split('/')[-1])
        else:
            print("loading:", root.split('/')[-2])
        self.img1_all, self.img2_all, self.mask1_all, self.mask2_all, self.corr_all, self.img1_list = ([] for i in range(6))
        for img0_filename in tqdm(self.img0_list):
            sheet_id = "_".join(img0_filename.split("/")[-1].split("_")[:-1])
            img1 = cv2.imread(os.path.join(root, f"{sheet_id}_0.png"))
            img2 = cv2.imread(os.path.join(root, f"{sheet_id}_1.png"))
            mask1 = cv2.imread(os.path.join(root, f"{sheet_id}_0_mask.png"), 0)
            mask2 = cv2.imread(os.path.join(root, f"{sheet_id}_1_mask.png"), 0)
            corr = np.load(os.path.join(root, f"{sheet_id}_corr.npy"))
            self.img1_all.append(img1)
            self.img2_all.append(img2)
            self.mask1_all.append(mask1 > 50)
            self.mask2_all.append(mask2 > 50)
            self.corr_all.append(corr)
            self.img1_list.append(os.path.join(root, f"{sheet_id}_0.png"))
        self.img1_all = np.stack(self.img1_all, axis=0)
        self.img2_all = np.stack(self.img2_all, axis=0)
        self.mask1_all = np.stack(self.mask1_all, axis=0)
        self.mask2_all = np.stack(self.mask2_all, axis=0)
        self.corr_all = np.stack(self.corr_all, axis=0)

    def __len__(self):
        return len(self.img1_list)

    def __getitem__(self, idx):
        img1 = self.img1_all[idx]
        img2 = self.img2_all[idx]
        mask1 = self.mask1_all[idx]
        mask2 = self.mask2_all[idx]
        corr = self.corr_all[idx]
        img1_filename = self.img1_list[idx].split('/')[-1]

        if self.color_jittering:
            brightness_factor = np.random.uniform(1 - 0.2, 1 + 0.2)
            contrast_factor = np.random.uniform(1 - 0.2, 1 + 0.2)
            saturation_factor = np.random.uniform(1 - 0.2, 1 + 0.2)
            hue_factor = np.random.uniform(-0.2, 0.2)

            img1 = T.ToTensor()(img1)
            img2 = T.ToTensor()(img2)
            img1 = T.functional.adjust_brightness(img1, brightness_factor)
            img1 = T.functional.adjust_contrast(img1, contrast_factor)
            img1 = T.functional.adjust_saturation(img1, saturation_factor)
            img1 = T.functional.adjust_hue(img1, hue_factor)
            img2 = T.functional.adjust_brightness(img2, brightness_factor)
            img2 = T.functional.adjust_contrast(img2, contrast_factor)
            img2 = T.functional.adjust_saturation(img2, saturation_factor)
            img2 = T.functional.adjust_hue(img2, hue_factor)
        else:
            img1 = T.ToTensor()(img1)
            img2 = T.ToTensor()(img2)

        return {"img1": img1, "img2": img2, "mask1": mask1, "mask2": mask2, "corr": corr, "img1_filename": img1_filename}
