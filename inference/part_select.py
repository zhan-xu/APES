import copy, os, time, glob, itertools, numpy as np, cv2
from scipy.io import savemat, loadmat
from scipy.optimize import linprog
from scipy.spatial import Delaunay
from skimage import transform
from skimage.measure import label, regionprops, ransac
from skimage.transform import SimilarityTransform
from inference.create_cage import create_cage
from inference.ARAP_RECON_fast import ARAP_Rendering_Deformer as Deformer
from utils.os_utils import mkdir_p


class PartSelector:
    def __init__(self, img_array=None, mask_array=None, part_database=None, corr_array=None, num_trial=None):
        """
        :param img_array: N, H, W, 3
        :param mask_array: N, H, W
        :param part_database: dict with "img_id" and "seg"
        :param corr_array: Dict, num_pair, H, W, 2
        :param num_trial:
        """
        self.K = 5
        self.paths_topk = []
        self.deformed_cages = []
        if img_array is not None:
            self.img_array = img_array
            self.num_frame = len(img_array)
        if mask_array is not None:
            self.mask_array = mask_array
        if corr_array is not None:
            self.corr_array = corr_array
        if part_database is not None:
            self.part_database = part_database
        if num_trial is not None:
            self.num_trial = num_trial

    def create_database(self):
        part_database_list = []
        for partition in self.part_database:
            pred_seg = partition["seg"]
            for l in range(pred_seg.max() + 1):
                mask_l = ((pred_seg == l) * 255).astype(np.uint8)
                if (mask_l > 0).sum() < 150:
                    continue
                # check if there is already same part from the same image
                part_unique = True
                for p in part_database_list:
                    if p["img_id"] == partition["img_id"] and (((p["mask"]>0) & (mask_l>0)).sum() / ((p["mask"]>0) | (mask_l>0)).sum()) > 0.97:
                        part_unique = False
                        break
                if part_unique:
                    part_database_list.append({"img_id": partition["img_id"], "mask": mask_l})  # "frequency": 1
        return part_database_list

    def save(self, output_folder):
        np.save(os.path.join(output_folder, "img_array.npy"), self.img_array)
        np.save(os.path.join(output_folder, "mask_array.npy"), self.mask_array)
        savemat(os.path.join(output_folder, f"corr_array.mat"), self.corr_array)
        # for i in range(len(self.part_database)):
        #     savemat(os.path.join(output_folder, f"part_{i}.mat"), self.part_database[i])
        print("save done")

    def load(self, snapshot_folder, num_trial, num_frame):
        self.num_trial = num_trial
        self.num_frame = num_frame
        self.img_array = np.load(os.path.join(snapshot_folder, "img_array.npy"))
        self.mask_array = np.load(os.path.join(snapshot_folder, "mask_array.npy"))
        self.corr_array = loadmat(os.path.join(snapshot_folder, f"corr_array.mat"))
        self.corr_array.pop("__header__")
        self.corr_array.pop("__version__")
        self.corr_array.pop("__globals__")
        self.part_database = []
        # part_matfiles = glob.glob(os.path.join(snapshot_folder, "part_*.mat"))
        # for i in range(len(part_matfiles)):
        #     part_i = loadmat(os.path.join(snapshot_folder, f"part_{i}.mat"))
        #     self.part_database.append({"img_id": part_i["img_id"][0, 0], "mask": part_i["mask"], "frequency": part_i["frequency"]})
        print("load done")

    def load_parts(self, part_folder, k=5):
        self.img_ids_topk = []
        self.masks_topk = []
        for i in range(k):
            img_ids_i = np.load(os.path.join(part_folder, f"masks_ids_{i}.npy"))
            mask_i = np.load(os.path.join(part_folder, f"masks_{i}.npy"))
            self.img_ids_topk.append(img_ids_i)
            self.masks_topk.append(mask_i)

    @staticmethod
    def overlap(mask1, mask2, thrd):
        intersect = (mask1 > 0) & (mask2 > 0)
        #iou = intersect.reshape(intersect.shape[0], -1).sum(-1) / (union.reshape(union.shape[0], -1).sum(-1) + 1e-6)
        '''if (mask_overlap_mask1.sum(axis=0).sum(axis=0) / ((mask1_small > 0).sum(axis=0).sum(axis=0) + 1e-6)).max() > thrd or \
                        (mask_overlap_mask2.sum(axis=0).sum(axis=0) / ((mask2_small > 0).sum(axis=0).sum(axis=0) + 1e-6)).max() > thrd:
                    # mask_overlap = mask_overlap.astype(np.uint8) * 255
                    # mask_overlap_mask1 = mask_overlap_mask1.astype(np.uint8) * 255
                    # mask_overlap_mask2 = mask_overlap_mask2.astype(np.uint8) * 255
                    # cv2.imshow("mask_in", np.concatenate((mask1, mask2, mask_overlap, mask_overlap_mask1, mask_overlap_mask2), axis=1))
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                    return True
                return False
                return iou.mean() > thrd'''
        if ((intersect.sum(-1).sum(-1)) / ((mask1>0).sum(-1).sum(-1) + 1e-6)).mean() > thrd or \
                ((intersect.sum(-1).sum(-1)) / ((mask2>0).sum(-1).sum(-1) + 1e-6)).mean() > thrd:
            return True
        else:
            return False


    @staticmethod
    def morphology_eligible(mask_in, area_thrd):
        # check area
        if np.sum(mask_in > 0) < area_thrd:
            return False
        # check connected components number
        # mask_in_dilate = cv2.dilate(mask_in, np.ones((3, 3), np.uint8), iterations=1)
        label_image = label(mask_in, connectivity=2)
        if label_image.max() > 1:
            return False
        return True

    @staticmethod
    def path_search_mc(start_hop, current_hops, compatibility_matrix, deformed_masks, eval_masks, banned_ids):
        banned_id = np.argwhere(compatibility_matrix[start_hop] == 0).squeeze(axis=1).tolist()
        banned_ids += banned_id
        next_choices = np.argwhere(compatibility_matrix[start_hop]).squeeze(axis=1)
        next_choices = [p for p in next_choices if p not in current_hops and p not in banned_ids]
        if len(next_choices) == 0:
            return current_hops
        next_hop = np.random.choice(next_choices)
        current_hops.append(int(next_hop))
        if PartSelector.cal_coverage(current_hops, deformed_masks, eval_masks) > 0.95:
            return current_hops
        else:
            PartSelector.path_search_mc(next_hop, current_hops, compatibility_matrix, deformed_masks, eval_masks, banned_ids)

    @staticmethod
    def cal_coverage(steps, deformed_masks, eval_masks):
        masked_steps = np.zeros_like(eval_masks)
        for step in steps:
            masked_steps = np.maximum(masked_steps, deformed_masks[step])

        # for i in range(masked_steps.shape[0]):
        #     cv2.imshow("mask", np.concatenate((masked_steps[i], eval_masks[i]), axis=1))
        #     print(((masked_steps[i] > 0) & (eval_masks[i] > 0)).sum() / (eval_masks[i] > 0).sum())
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()

        overlap = (masked_steps > 0) * (eval_masks > 0)
        coverage = (overlap.reshape(overlap.shape[0], -1).sum(1) / ((eval_masks > 0).reshape(eval_masks.shape[0], -1).sum(1)+1e-6)).mean()
        # visualize_mask_step(masked_steps)
        return coverage

    def deform_parts_to_all(self):
        """
        self.corr_array: Dict, num_pair, H, W, 2
        vis_in: Dict, num_pair, H, W
        """
        deformed_parts_all = []
        deformed_masks_all = []
        deformed_cages_all = []
        deformed_params_all = []
        img_ids_all = []
        for ii, partition in enumerate(self.part_database):
            img_id = partition["img_id"]
            mask_part = partition["mask"]
            img_part = np.where(mask_part[..., None].repeat(3, axis=-1), self.img_array[img_id], 255*np.ones_like(self.img_array[img_id]))
            deformed_imgs_from_this_part = []
            deformed_masks_from_this_part = []
            deformed_params_from_this_part = []
            for eval_img_id in range(self.num_frame):
                if eval_img_id == img_id:
                    img_warp = copy.deepcopy(img_part)
                    mask_warp = copy.deepcopy(mask_part)
                    # cv2.imshow("warp masks", mask_part)
                    # cv2.imshow("warp part", img_warp)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                    _, cage_verts = create_cage(img_warp, mask_warp, type="obb_grid")
                    deformed_cages_all.append(cage_verts)
                    deformed_params_from_this_part.append(np.eye(3))
                    # visualize_cage(deformed_masks_pid[eval_img_id], cage_p_i, tri_face=tri_vis.simplices, show=True)
                else:
                    corr = self.corr_array[f"{img_id}{eval_img_id}"]
                    pts_src = np.argwhere(mask_part)
                    pts_tar = corr[pts_src[:, 0], pts_src[:, 1], 0:2]

                    rtran = transform.estimate_transform('similarity', pts_src, pts_tar)
                    deformed_params_from_this_part.append(rtran.params)

                    pts_tar_grid = np.meshgrid(np.arange(corr.shape[0]), np.arange(corr.shape[1]))
                    pts_tar_grid = np.stack((pts_tar_grid[1], pts_tar_grid[0]), axis=-1)
                    pts_tar_grid = pts_tar_grid.reshape(-1, 2)

                    pts_backwarp = rtran.inverse(pts_tar_grid)
                    pts_backwarp = np.round(pts_backwarp).astype(np.int64)
                    pts_backwarp = np.clip(pts_backwarp, 0, 255)

                    # back warp
                    backwarp_map = -np.ones((2, corr.shape[0], corr.shape[1]), dtype=np.int32)
                    backwarp_map[:, pts_tar_grid[:, 0], pts_tar_grid[:, 1]] = pts_backwarp.T
                    img_warp = np.stack([transform.warp(img_part[:, :, chn]/255.0, backwarp_map, cval=1.0) for chn in range(3)], axis=-1)
                    img_warp = np.round(img_warp * 255).astype(np.uint8)
                    mask_warp = transform.warp(mask_part/255.0, backwarp_map, cval=0.0)
                    mask_warp = np.round(mask_warp * 255).astype(np.uint8)
                    _, mask_warp = cv2.threshold(mask_warp, 50, 255, cv2.THRESH_BINARY)
                    img_warp = np.where(np.repeat(mask_warp[..., None], 3, axis=-1), img_warp, np.ones_like(img_warp)*255)
                    # cv2.imshow("warp_img", np.concatenate((img_part, self.img_array[eval_img_id], img_warp), axis=1))
                    # cv2.imshow("warp_mask", np.concatenate((mask_part, (self.mask_array[eval_img_id] * 255).astype(np.uint8), mask_warp), axis=1))
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                deformed_imgs_from_this_part.append(img_warp)
                deformed_masks_from_this_part.append(mask_warp)
            # cv2.imshow("deform_parts",
            #            np.concatenate((np.concatenate(deformed_imgs_from_this_part, axis=1),
            #                            np.concatenate(list(self.img_array), axis=1)), axis=0))
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            deformed_parts_all.append(np.stack(deformed_imgs_from_this_part, axis=0))
            deformed_masks_all.append(np.stack(deformed_masks_from_this_part, axis=0))
            deformed_params_all.append(np.stack(deformed_params_from_this_part, axis=0))
            img_ids_all.append(img_id)
        deformed_parts_all = np.stack(deformed_parts_all, axis=0)
        deformed_masks_all = np.stack(deformed_masks_all, axis=0)
        deformed_params_all = np.stack(deformed_params_all, axis=0)
        self.deformed_parts = deformed_parts_all
        self.deformed_masks = deformed_masks_all
        self.deformed_params = deformed_params_all
        self.deformed_cages = deformed_cages_all
        self.img_ids = np.array(img_ids_all)

    def reconstruct(self, debug_folder):
        recon_loss = []
        parts_img_warped_all = []
        parts_mask_warped_all = []
        mesh_V_all = []
        mesh_F_all = []
        for k in range(len(self.paths_topk)):
            print(f"Optimizing path {k + 1} / 5.")
            img_recon, img_parts_warp, mask_parts_warp, mesh_V, mesh_F = self.reconstruct_one_subset(self.paths_topk[k])
            recon_loss.append(np.sum((img_recon / 255.0 - self.img_array / 255.0) ** 2, axis=-1).mean())
            parts_img_warped_all.append(copy.deepcopy(img_parts_warp))
            parts_mask_warped_all.append(copy.deepcopy(mask_parts_warp))
            mesh_V_all.append(copy.deepcopy(mesh_V))
            mesh_F_all.append(copy.deepcopy(mesh_F))
            # for i in range(len(self.img_array)):
            #     cv2.imwrite(os.path.join(debug_folder, f"recon_{k}_{i}.png"),
            #                 np.concatenate((self.img_array[i], img_recon[i]), axis=1))
        best_k = np.argmin(np.array(recon_loss))
        print(f"best_k: {best_k}, recon_loss: {recon_loss[best_k]}")
        output_imgs, output_mask = [], []
        for pid in self.paths_topk[best_k]:
            output_mask.append(self.part_database[pid]["mask"])
            output_imgs.append(self.img_array[self.part_database[pid]["img_id"]])
        return np.array(output_imgs), np.stack(output_mask, axis=0), parts_img_warped_all[best_k], parts_mask_warped_all[best_k]

    def reconstruct_one_subset(self, parts_ids):
        shape = self.img_array.shape
        #recon_imgs_naive = np.ones_like(self.img_array) * 255
        src_part_all = np.ones((len(parts_ids), shape[1], shape[2], shape[3]), dtype=np.uint8) * 255
        src_mask_all = np.zeros((len(parts_ids), shape[1], shape[2]), dtype=np.uint8) * 255
        src_anchor_shift = [[] for _ in range(len(self.img_array))]
        src_anchor_list = []
        src_face_list = []
        for step_i, pid in enumerate(parts_ids):
            deformed_parts_pid = self.deformed_parts[pid]
            deformed_masks_pid = self.deformed_masks[pid]
            cage_pid = self.deformed_cages[pid]
            tri_pid = Delaunay(cage_pid)
            for eval_img_id in range(self.img_array.shape[0]):
                # naive paste
                # recon_imgs_naive[eval_img_id] = \
                #     np.where(np.repeat(deformed_masks_pid[eval_img_id,:,:, None], 3, axis=-1),
                #              deformed_parts_pid[eval_img_id], recon_imgs_naive[eval_img_id])

                # optimization
                #cage_p_i = create_cage(deformed_parts_pid[eval_img_id], deformed_masks_pid[eval_img_id], type="obb_grid")
                rtran = SimilarityTransform(matrix=self.deformed_params[pid, eval_img_id])
                cage_p_i = rtran(cage_pid)
                #visualize_cage(deformed_masks_pid[eval_img_id], np.clip(np.round(cage_p_i), 0, 255).astype(np.int64), tri_face=tri_pid.simplices, show=True)
                #src_part_all[eval_img_id, step_i] = deformed_parts_pid[eval_img_id]
                #src_mask_all[eval_img_id, step_i] = deformed_masks_pid[eval_img_id]
                src_anchor_shift[eval_img_id].append(((cage_p_i - cage_pid) / shape[1]).astype(np.float32))
                #src_face_list[eval_img_id].append(tri_pid.simplices)
            src_part_all[step_i] = deformed_parts_pid[self.img_ids[pid]]
            src_mask_all[step_i] = deformed_masks_pid[self.img_ids[pid]]
            src_anchor_list.append(cage_pid / shape[1])
            src_face_list.append(tri_pid.simplices)

        deformer = Deformer(src_part_all / 255.0, src_mask_all / 255.0, self.img_array / 255.0, src_face_list, src_anchor_list, src_anchor_shift)
        recon_imgs, img_parts_warp, mask_parts_warp, mesh_V, mesh_F = deformer.run(lr=0.001, num_iter=80, w_reg=0.05) #lr=0.001, num_iter=80, w_reg=0.05
        # for i in range(len(self.img_array)):
        #     cv2.imwrite(f"/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/results/select/yoga/debug/{i}_cage_rec.png",
        #                 np.concatenate((self.img_array[i], recon_imgs[i]), axis=1))
        return recon_imgs, img_parts_warp, mask_parts_warp, mesh_V, mesh_F

    def filter_parts(self, filter_thrd):
        iou_matrix = np.zeros((len(self.deformed_parts), len(self.deformed_parts)))
        for i in np.arange(len(self.deformed_parts)):
            for j in np.arange(i+1, len(self.deformed_parts)):
                intersect = ((self.deformed_masks[i]) > 0) & (self.deformed_masks[j] > 0)
                overlap = (self.deformed_masks[i] > 0) | (self.deformed_masks[j] > 0)
                iou_matrix[i, j] = (intersect.sum(-1).sum(-1) / (overlap.sum(-1).sum(-1) + 1e-8)).mean()

        iou_matrix = iou_matrix + iou_matrix.T
        iou_matrix += np.eye(len(self.deformed_parts))
        iou_matrix_bin = (iou_matrix > filter_thrd)
        valid_ids = np.argwhere(iou_matrix_bin.sum(axis=1) >= 3).squeeze(1)
        self.part_database = [self.part_database[i] for i in valid_ids]
        self.deformed_parts = self.deformed_parts[valid_ids]
        self.deformed_masks = self.deformed_masks[valid_ids]
        self.deformed_cages = [self.deformed_cages[i] for i in valid_ids]
        self.deformed_params = self.deformed_params[valid_ids]
        self.img_ids = self.img_ids[valid_ids]
        self.frequency = iou_matrix[valid_ids, :][:, valid_ids].sum(0)
        return iou_matrix[valid_ids, :][:, valid_ids]

    def gen_compatibility_matrix(self, deformed_masks, overlap_thrd):
        compatibility_matrix = np.zeros((len(deformed_masks), len(deformed_masks)), dtype=np.uint8)
        for pi in range(len(deformed_masks)):
            for pj in range(pi + 1, len(deformed_masks)):
                # if pi == 0 and pj == 10:
                #     cv2.imshow("debug", np.concatenate((deformed_masks[pi, 0], deformed_masks[pj, 0]), axis=0))
                #     cv2.waitKey()
                #     cv2.destroyAllWindows()
                compatibility_matrix[pi, pj] = 1 - PartSelector.overlap(deformed_masks[pi], deformed_masks[pj], thrd=overlap_thrd)
        compatibility_matrix = np.maximum(compatibility_matrix, compatibility_matrix.T)
        return compatibility_matrix

    def evaluate_paths(self, all_paths):
        all_mse_loss = []
        for p_id, path in enumerate(all_paths):
            recons_eval_img = np.ones_like(self.img_array, dtype=np.uint8) * 255
            for step in path:
                part_step = self.deformed_parts[step]
                mask_step = self.deformed_masks[step]
                recons_eval_img = np.where(np.repeat(mask_step[..., np.newaxis], 3, axis=-1), part_step, recons_eval_img)
            # cv2.imshow("recons_eval_img", np.concatenate((recons_eval_img, eval_imgs[eval_id]), axis=1))
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            mse_loss = ((recons_eval_img / 255.0 - self.img_array / 255.0) ** 2).sum(-1).mean()
            all_mse_loss.append(mse_loss / len(self.img_array))
        all_mse_loss = np.array(all_mse_loss)
        all_paths_reorder = [all_paths[i] for i in np.argsort(all_mse_loss)]
        return all_paths_reorder

    def show_selection(self, path, img_array, show):
        img_show = []
        for step in path:
            img_p = img_array[self.img_ids[step]].copy()
            deformed_mask = self.deformed_masks[step, self.img_ids[step]]
            pix_pos = np.argwhere(deformed_mask)
            mask_red = np.ones_like(img_p) * 255
            mask_red[pix_pos[:, 0], pix_pos[:, 1]] = np.array([150, 150, 40])
            mask_red = cv2.GaussianBlur(mask_red, (3, 3), cv2.BORDER_DEFAULT)
            img_show_step = cv2.addWeighted(img_p, 0.3, mask_red, 0.7, 0)
            img_show.append(img_show_step)
        img_show = np.concatenate(img_show, axis=1)
        if show:
            cv2.imshow("parts", img_show)
            cv2.waitKey()
            cv2.destroyAllWindows()
        return img_show

    def select(self, slic_list, filter_thrd, debug_folder=None):
        print("number of total parts:", len(self.part_database))
        self.deform_parts_to_all()
        print("Part deformation finished.")
        sim_mat = self.filter_parts(filter_thrd=filter_thrd)
        print("Filtering parts finished.")
        print("number of total parts:", len(self.part_database))

        # mkdir_p(os.path.join(debug_folder, "parts/"))
        # for i in range(len(self.part_database)):
        #     savemat(os.path.join(debug_folder, f"parts/part_{i}.mat"), self.part_database[i])
        # for i in range(len(self.deformed_cages)):
        #     np.save(os.path.join(debug_folder, f"parts/cage_{i}.npy"), self.deformed_cages[i])
        # np.save(os.path.join(debug_folder, f"parts/deformed_parts.npy"), self.deformed_parts)
        # np.save(os.path.join(debug_folder, f"parts/deformed_masks.npy"), self.deformed_masks)
        # np.save(os.path.join(debug_folder, f"parts/deformed_params.npy"), self.deformed_params)
        # np.save(os.path.join(debug_folder, f"parts/deformed_parts_imgids.npy"), self.img_ids)
        # np.save(os.path.join(debug_folder, f"parts/sim_mat.npy"), sim_mat)
        #
        # self.deformed_parts = np.load(os.path.join(debug_folder, f"parts/deformed_parts.npy"))
        # self.deformed_masks = np.load(os.path.join(debug_folder, f"parts/deformed_masks.npy"))
        # self.deformed_params = np.load(os.path.join(debug_folder, f"parts/deformed_params.npy"))
        # self.img_ids = np.load(os.path.join(debug_folder, f"parts/deformed_parts_imgids.npy"))
        # sim_mat = np.load(os.path.join(debug_folder, f"parts/sim_mat.npy"))
        # part_matfiles = glob.glob(os.path.join(debug_folder, "parts/part_*.mat"))
        # for i in range(len(part_matfiles)):
        #     part_i = loadmat(os.path.join(debug_folder, f"parts/part_{i}.mat"))
        #     self.part_database.append({"img_id": part_i["img_id"][0, 0], "mask": part_i["mask"]})
        # for i in range(len(self.img_ids)):
        #     self.deformed_cages.append(np.load(os.path.join(debug_folder, f"parts/cage_{i}.npy")))

        # for i, part_item in enumerate(self.part_database):
        #     cv2.imwrite(os.path.join(debug_folder, f"part_{i}.png"), np.where(part_item["mask"][..., None], self.img_array[part_item["img_id"]], np.ones_like(self.img_array[part_item["img_id"]])*255))

        # linear programing
        cover_names = []
        cover_all = []
        for t in range(len(slic_list)):
            for l in range(slic_list[t].max() + 1):
                pos_l = np.argwhere(slic_list[t] == l)
                cover_tl = (self.deformed_masks[:, t, pos_l[:, 0], pos_l[:, 1]] / 255.0).sum(-1) > 0.7 * len(pos_l)
                cover_names.append((t, l))
                cover_all.append(cover_tl)
        A = np.stack(cover_all, axis=0).astype(np.float64)
        valid_ids = np.argwhere(A.sum(1) >= 1).squeeze(axis=1)
        A = A[valid_ids]
        cover_names = [cover_names[i] for i in valid_ids]
        b = np.ones(len(cover_names))
        c = np.ones(len(self.deformed_masks))
        lp_opt = linprog(c=c, A_ub=-A, b_ub=-b, bounds=(0.0, 1.0), method='interior-point')  # interior-point, highs-ds
        prob_sample = lp_opt.x

        part_IDs = []
        part_probs = []
        for pid in range(len(prob_sample)):
            if prob_sample[pid] > 1e-2:
                part_IDs_nbrs = []
                for pid_nbr in np.argsort(sim_mat[pid])[::-1][0:6]:
                    part_IDs_nbrs.append(pid_nbr)
                part_IDs.append(np.array(part_IDs_nbrs))
                part_probs.append(prob_sample[pid])
        part_IDs = np.array(part_IDs)
        part_probs = np.array(part_probs)

        compatible_matrix = self.gen_compatibility_matrix(self.deformed_masks, overlap_thrd=0.2)
        all_paths = []
        for trial_i in range(self.num_trial):
            path = []
            num_iter = 0
            while PartSelector.cal_coverage(path, self.deformed_masks, (self.mask_array * 255).astype(np.uint8)) <= 0.95:
                num_iter += 1
                if num_iter > 50:
                    break
                tossing = np.random.uniform(size=len(part_IDs))
                steps_this_toss = np.argwhere(tossing <= part_probs).squeeze(1)
                steps_this_toss = steps_this_toss[np.argsort(part_probs[steps_this_toss])[::-1]]
                steps_this_toss = part_IDs[steps_this_toss]
                for step_i in range(len(steps_this_toss)):
                    step = np.random.choice(steps_this_toss[step_i])
                    if len(compatible_matrix[path, step]) == 0 or np.all(compatible_matrix[path, step]) == 1:
                           path.append(step)
                steps_not_selected = [p for p in part_IDs.flatten() if p not in path]
                if np.sum(compatible_matrix[path, :][:, steps_not_selected].prod(axis=0)) == 0:  # no part can be chosen
                    break
            all_paths.append(path)
        # reference: https://stackoverflow.com/questions/2213923/removing-duplicates-from-a-list-of-lists
        all_paths.sort()
        all_paths = list(all_paths for all_paths, _ in itertools.groupby(all_paths))

        # evaluate valid paths
        all_path_reorder = self.evaluate_paths(all_paths)
        print("Path seach and evaluate finished.")

        if debug_folder is not None:
            for i in range(self.K):
                if i >= len(all_path_reorder):
                    continue
                print(all_path_reorder[i])
                img = self.show_selection(all_path_reorder[i], self.img_array, show=False)
                cv2.imwrite(os.path.join(debug_folder, f"selection_{i}.png"), img)
                self.paths_topk.append(all_path_reorder[i])
        return self.paths_topk

