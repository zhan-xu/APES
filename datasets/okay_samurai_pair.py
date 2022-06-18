import glob, os, h5py, cv2, numpy as np, torch, torchvision.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm


class OkaySamuraiPair(Dataset):
    """
    val: chief, xbox, pc, seth, filmbot
    test: switch, ramirez, play station, hackbot, suzanne
    """
    def __init__(self, root, train_corr, color_jittering=False):
        self.root = root
        self.train_corr = train_corr
        self.color_jittering = color_jittering
        if len(root.split('/')[-1]) > 0:
            split_name = root.split('/')[-1]
        else:
            split_name = root.split('/')[-2]
        print("loading:", split_name)
        hf = h5py.File(os.path.join(self.root, f'okaysamurai_{split_name}.h5'), 'r')
        self.img1_all, self.img2_all, self.mask1_all, self.mask2_all, self.seg1_all, self.seg2_all, self.corr_all, \
        self.slic_all, self.slic_label_all, self.src_pixel_group_all, self.img1_list, self.pred_corr_all= ([] for i in range(12))
        for sheet_id in tqdm(range(hf[f"img"].shape[0])): #hf[f"img"].shape[0]
            img1, img2 = np.split(hf["img"][sheet_id], 2, axis=-1)
            mask1, mask2 = np.split(hf["mask"][sheet_id], 2, axis=-1)
            seg1, seg2 = np.split(hf["seg"][sheet_id], 2, axis=-1)
            slic_map1, slic_map2 = np.split(hf["slic"][sheet_id], 2, axis=-1)
            slic_segID1, slic_segID2 = np.split(hf["slic_segID"][sheet_id], 2, axis=-1)
            src_pixel_group1, src_pixel_group2 = np.split(hf["src_pixel_group"][sheet_id], 2, axis=-1)
            corr = hf["corr"][sheet_id]
            mask1, mask2, seg1, seg2, slic_map1, slic_map2, slic_segID1, slic_segID2 = \
                mask1.squeeze(-1), mask2.squeeze(-1), seg1.squeeze(-1), seg2.squeeze(-1), \
                slic_map1.squeeze(-1), slic_map2.squeeze(-1), slic_segID1.squeeze(-1), slic_segID2.squeeze(-1)

            self.img1_all.append(img1)
            self.img2_all.append(img2)
            self.mask1_all.append(mask1.astype(bool))
            self.mask2_all.append(mask2.astype(bool))
            self.corr_all.append(corr)
            self.seg1_all.append(seg1)
            self.seg2_all.append(seg2)
            self.slic_all.append(np.stack((slic_map1, slic_map2), axis=-1))
            self.slic_label_all.append(np.stack((slic_segID1, slic_segID2), axis=-1))
            self.src_pixel_group_all.append(np.concatenate((src_pixel_group1, src_pixel_group2), axis=-1))
            self.img1_list.append(hf["names"][sheet_id].decode())

            if not self.train_corr:
                data_name = hf["names"][sheet_id].decode()
                pred_corr = np.load(os.path.join(self.root, f"{data_name}_predcorr.npy"))
                self.pred_corr_all.append(pred_corr)

        self.img1_all = np.stack(self.img1_all, axis=0)
        self.img2_all = np.stack(self.img2_all, axis=0)
        self.mask1_all = np.stack(self.mask1_all, axis=0)
        self.mask2_all = np.stack(self.mask2_all, axis=0)
        self.corr_all = np.stack(self.corr_all, axis=0)
        self.seg1_all = np.stack(self.seg1_all, axis=0)
        self.seg2_all = np.stack(self.seg2_all, axis=0)
        self.slic_all = np.stack(self.slic_all, axis=0)
        self.slic_label_all = np.stack(self.slic_label_all, axis=0)
        self.src_pixel_group_all = np.stack(self.src_pixel_group_all, axis=0)
        if not self.train_corr:
            self.pred_corr_all = np.stack(self.pred_corr_all, axis=0)

    def __len__(self):
        return len(self.img1_list)

    def __getitem__(self, idx):
        img1 = self.img1_all[idx]
        img2 = self.img2_all[idx]
        mask1 = self.mask1_all[idx]
        mask2 = self.mask2_all[idx]
        seg1 = self.seg1_all[idx]
        seg2 = self.seg2_all[idx]
        corr = self.corr_all[idx]
        img1_filename = self.img1_list[idx].split('/')[-1]
        if not self.train_corr:
            slic = self.slic_all[idx]
            slic_label = self.slic_label_all[idx]
            src_pixel_group = self.src_pixel_group_all[idx]

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

        if self.train_corr:
            return {"img1": img1, "img2": img2, "mask1": mask1, "mask2": mask2, "corr": corr, "img1_filename": img1_filename}
        else:
            pred_corr = self.pred_corr_all[idx]
            return {"img1": img1, "img2": img2, "mask1": mask1, "mask2": mask2, "corr": corr, "slic": slic,
                    "slic_label": slic_label, "seg1": seg1, "seg2": seg2, "src_pixel_group": src_pixel_group,
                    "img1_filename": img1_filename, "pred_corr": pred_corr}
