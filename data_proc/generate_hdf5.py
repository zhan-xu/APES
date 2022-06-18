import sys
sys.path.append("./")
import os, glob, h5py, numpy as np, json, cv2
from tqdm import tqdm
from utils.vis_utils import visualize_corr, visualize_seg
from skimage import segmentation

RES_DIM = 256


def generate_hd5():
    root_folder = "/mnt/DATA_LINUX/zhan/puppets/okay_samurai/"
    for split_name in ["train", "val", "test"]:
        filelist = glob.glob(os.path.join(root_folder, f"{split_name}/*_0.png"))

        hf = h5py.File(os.path.join(root_folder, f'{split_name}/okaysamural_{split_name}.h5'), 'w')
        hf.create_dataset('img', (len(filelist), RES_DIM, RES_DIM, 6), np.uint8)
        hf.create_dataset('mask', (len(filelist), RES_DIM, RES_DIM, 2), np.uint8)
        hf.create_dataset('seg', (len(filelist), RES_DIM, RES_DIM, 2), np.int16)
        hf.create_dataset('slic', (len(filelist), RES_DIM, RES_DIM, 2), np.int16)
        hf.create_dataset('slic_segID', (len(filelist), 50, 2), np.int16)
        hf.create_dataset('src_pixel_group', (len(filelist), 50, 120, 4), np.float32)
        hf.create_dataset('corr', (len(filelist), RES_DIM, RES_DIM, 4), np.int16)
        names = []

        for id, img0_filename in enumerate(tqdm(filelist)):
            name = "_".join(img0_filename.split("/")[-1].split("_")[0:2])
            img0 = cv2.imread(img0_filename)
            img1 = cv2.imread(img0_filename.replace("_0.png", "_1.png"))
            mask0 = cv2.imread(img0_filename.replace("_0.png", "_0_mask.png"), 0)
            mask1 = cv2.imread(img0_filename.replace("_0.png", "_1_mask.png"), 0)
            seg0 = np.load(img0_filename.replace("_0.png", "_0_seg.npy"))
            seg1 = np.load(img0_filename.replace("_0.png", "_1_seg.npy"))
            slic0 = np.load(img0_filename.replace("_0.png", "_0_slic.npy"))
            slic1 = np.load(img0_filename.replace("_0.png", "_1_slic.npy"))
            slic_segID0 = np.load(img0_filename.replace("_0.png", "_0_slic_segID.npy"))
            slic_segID1 = np.load(img0_filename.replace("_0.png", "_1_slic_segID.npy"))
            src_pixel_group0 = np.load(img0_filename.replace("_0.png", "_0_src_pixel_group.npy"))
            src_pixel_group1 = np.load(img0_filename.replace("_0.png", "_1_src_pixel_group.npy"))
            corr = np.load(img0_filename.replace("_0.png", "_corr.npy"))

            hf["img"][id] = np.concatenate((img0, img1), axis=-1)
            hf["mask"][id] = np.stack((mask0, mask1), axis=-1)
            hf["seg"][id] = np.stack((seg0, seg1), axis=-1)
            hf["slic"][id] = np.stack((slic0, slic1), axis=-1)
            hf["slic_segID"][id] = np.stack((slic_segID0, slic_segID1), axis=-1)
            hf["src_pixel_group"][id] = np.concatenate((src_pixel_group0, src_pixel_group1), axis=-1)
            hf["corr"][id] = corr
            names.append(name)

        string_dt = h5py.special_dtype(vlen=str)
        hf.create_dataset('names', data=names, dtype=string_dt)
        hf.close()


def visualize_hd5():
    split_name = "test"
    root_folder = os.path.join("/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/", f"{split_name}")
    hf = h5py.File(os.path.join(root_folder, f'okaysamural_{split_name}.h5'), 'r')
    while True:
        num_pair = hf[f"img"].shape[0]
        id = np.random.choice(num_pair, 1)[0]
        img0, img1 = np.split(hf["img"][id], 2, axis=-1)
        mask0, mask1 = np.split(hf["mask"][id], 2, axis=-1)
        seg0, seg1 = np.split(hf["seg"][id], 2, axis=-1)
        slic0, slic1 = np.split(hf["slic"][id], 2, axis=-1)
        slic_segID0, slic_segID1 = np.split(hf["slic_segID"][id], 2, axis=-1)
        src_pixel_group0, src_pixel_group1 = np.split(hf["src_pixel_group"][id], 2, axis=-1)
        corr = hf["corr"][id]
        mask0, mask1, seg0, seg1, slic0, slic1, slic_segID0, slic_segID1 = \
            mask0.squeeze(-1), mask1.squeeze(-1), seg0.squeeze(-1), seg1.squeeze(-1), \
            slic0.squeeze(-1), slic1.squeeze(-1), slic_segID0.squeeze(-1), slic_segID1.squeeze(-1)

        img_seg0 = visualize_seg(img0, seg0, show=False)
        img_seg1 = visualize_seg(img1, seg1, show=False)
        img_slic0 = segmentation.mark_boundaries(img0, slic0)
        img_slic0 = np.round(img_slic0 * 255).astype(np.uint8)
        img_slic1 = segmentation.mark_boundaries(img1, slic1)
        img_slic1 = np.round(img_slic1 * 255).astype(np.uint8)
        img_gtcorr_f = visualize_corr(img0, mask0, img1, corr[..., :2], show=False)
        img_gtcorr_b = visualize_corr(img1, mask1, img0, corr[..., 2:], show=False)

        slic_seg0 = -np.ones_like(seg0)
        slic_seg1 = -np.ones_like(seg1)
        for l in range(slic0.max() + 1):
            slic_seg0 = np.where(slic0 == l, slic_segID0[l], slic_seg0)
            slic_seg1 = np.where(slic1 == l, slic_segID1[l], slic_seg1)
        img_slic_seg0 = visualize_seg(img0, slic_seg0, show=False)
        img_slic_seg1 = visualize_seg(img1, slic_seg1, show=False)

        cv2.imshow("corr", np.concatenate((img_gtcorr_f, img_gtcorr_b), axis=1))
        cv2.imshow("slic", np.concatenate((img_seg0, img_seg1, img_slic0, img_slic1), axis=1))
        cv2.imshow("slic_seg", np.concatenate((img_slic_seg0, img_slic_seg1), axis=1))

        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_hd5()
    #visualize_hd5()
