import numpy as np, cv2, time
from collections import deque
from scipy.spatial import Delaunay
import scipy.sparse as sparse
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesUV
)
from pytorch3d.renderer.blending import BlendParams, hard_rgb_blend
from torch_batch_svd import svd

RES_DIM = 256.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimpleShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image


class MovingAverage(object):
    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.queue = deque(maxlen=size)

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        self.queue.append(val)
        return sum(self.queue) / len(self.queue)


def build_cotan_laplacian( points, tris):
    a,b,c = (tris[:,0],tris[:,1],tris[:,2])
    # A = np.take( points, a, axis=1 )
    # B = np.take( points, b, axis=1 )
    # C = np.take( points, c, axis=1 )

    # eab,ebc,eca = (B-A, C-B, A-C)
    # eab = eab/np.linalg.norm(eab,axis=0)[None,:]
    # ebc = ebc/np.linalg.norm(ebc,axis=0)[None,:]
    # eca = eca/np.linalg.norm(eca,axis=0)[None,:]
    #
    # alpha = np.arccos( -np.sum(eca*eab,axis=0) )
    # beta  = np.arccos( -np.sum(eab*ebc,axis=0) )
    # gamma = np.arccos( -np.sum(ebc*eca,axis=0) )
    #
    # wab,wbc,wca = ( 1.0/(np.tan(gamma)+1e-6), 1.0/(np.tan(alpha)+1e-6), 1.0/(np.tan(beta)+1e-6) )
    # rows = np.concatenate((   a,   b,   a,   b,   b,   c,   b,   c,   c,   a,   c,   a ), axis=0 )
    # cols = np.concatenate((   a,   b,   b,   a,   b,   c,   c,   b,   c,   a,   a,   c ), axis=0 )
    # vals = np.concatenate(( wab, wab,-wab,-wab, wbc, wbc,-wbc,-wbc, wca, wca,-wca, -wca), axis=0 )
    rows = np.concatenate((a, a, b, b, c, c, a, b, b, c, c, a), axis=0)
    cols = np.concatenate((a, a, b, b, c, c, b, a, c, b, a, c), axis=0)
    vals = np.concatenate((np.ones(6*len(tris), dtype=np.float32), -np.ones(6*len(tris), dtype=np.float32)), axis=0)
    L_numpy = sparse.coo_matrix((vals,(rows,cols)),shape=(points.shape[1],points.shape[1]), dtype=float).tocsc()
    L_tensor = torch.sparse_coo_tensor(torch.stack((torch.from_numpy(rows), torch.from_numpy(cols)), dim=0), torch.from_numpy(vals), (points.shape[1], points.shape[1]))
    return L_numpy, L_tensor.to(device)

def build_weights_and_adjacency( points, tris):
    L_numpy, L_tensor = build_cotan_laplacian( points, tris )
    n_pnts, n_nbrs = (points.shape[1], L_numpy.getnnz(axis=0).max()-1)
    nbrs = np.ones((n_pnts,n_nbrs),dtype=int)*np.arange(n_pnts,dtype=int)[:,None]
    wgts = np.zeros((n_pnts,n_nbrs),dtype=float)

    for idx,col in enumerate(L_numpy):
        msk = col.indices != idx
        indices = col.indices[msk]
        values  = col.data[msk]
        nbrs[idx,:len(indices)] = indices
        wgts[idx,:len(indices)] = -values
        #wgts[idx, :len(indices)] = 1.0

    return nbrs, wgts, L_tensor

class ARAP_Rendering_Deformer:
    def __init__(self, img_parts, mask_parts, target_imgs, faces, src_anchor, anchor_shift_init):
        """
        Args:
            img_parts: P, H, W, 3  range (0-1)
            mask_parts: P, H, W  range (0-1)
            target_imgs:  I, H, W, 3  range (0-1)
            faces: list P x [Tri x 3]
            src_anchor: list P x [Anc x 2]
            anchor_shift_init:  list I x list P x [Anc_p x 2]
        """
        # convert all input to tensor if they are numpy array
        if isinstance(src_anchor[0], np.ndarray):
            for p in range(len(src_anchor)):
                src_anchor[p] = torch.from_numpy(src_anchor[p]).float().to(device)
        if isinstance(anchor_shift_init[0][0], np.ndarray):
            for i in range(len(anchor_shift_init)):
                for pid in range(len(anchor_shift_init[i])):
                    anchor_shift_init[i][pid] = torch.from_numpy(anchor_shift_init[i][pid]).to(device)
        if isinstance(target_imgs, np.ndarray):
            target_imgs = torch.from_numpy(target_imgs).float().to(device)
        if isinstance(faces[0], np.ndarray):
            for pid in range(len(faces)):
                faces[pid] = torch.from_numpy(faces[pid]).long().to(device)
        if isinstance(img_parts, np.ndarray):
            img_parts = torch.from_numpy(img_parts).float().to(device)
        if isinstance(mask_parts, np.ndarray):
            mask_parts = 1 - torch.from_numpy(mask_parts).float().unsqueeze(dim=-1).repeat(1,1,1,3).to(device)

        # initialize in-class members
        self.pts, self.tris = [], []
        cumsum_i = 0
        for i in range(len(target_imgs)):
            cumsum_p = 0
            cum_pts, cum_face = [], []
            for p in range(len(src_anchor)):
                cum_face.append(faces[p] + cumsum_p)
                cumsum_p += len(src_anchor[p])
                cum_pts.append(src_anchor[p] + anchor_shift_init[i][p])
            self.pts.append(torch.cat(cum_pts, dim=0))
            self.tris.append(torch.cat(cum_face, dim=0) + cumsum_i)
            cumsum_i += cumsum_p
        self.pts = torch.cat(self.pts, dim=0)
        self.tris = torch.cat(self.tris, dim=0)

        # import matplotlib.tri as mtri
        # import matplotlib.pyplot as plt
        # triangulation = mtri.Triangulation(self.pts[:,0].to("cpu").numpy(), self.pts[:,1].to("cpu").numpy(), self.tris.to("cpu").numpy())
        # plt.scatter(self.pts[:,0].to("cpu").numpy(), self.pts[:,1].to("cpu").numpy(), color='red')
        # plt.triplot(triangulation, 'g-h')
        # plt.show()

        self.src_anchor = src_anchor
        self.anchor_shift_init = anchor_shift_init
        self.target_imgs = target_imgs
        self.img_parts = img_parts
        self.mask_parts = mask_parts
        self.faces = faces
        self._nbrs, wgts, self.L = build_weights_and_adjacency(self.pts.to("cpu").numpy().T, self.tris.to("cpu").numpy())
        self._wgts = torch.from_numpy(wgts).float().to(device)

        self.texture_img, self.texture_mask = [], []
        for p in range(len(src_anchor)):
            texture_img = ARAP_Rendering_Deformer.set_texture(img_parts[p], src_anchor[p], faces[p])
            texture_mask = ARAP_Rendering_Deformer.set_texture(mask_parts[p], src_anchor[p], faces[p])
            self.texture_img.append(texture_img)
            self.texture_mask.append(texture_mask)

    @staticmethod
    def set_texture(textmap, cage_src, face_vid):
        texture_image = textmap[None, ...]  # (1, H, W, 3)
        verts_uvs = torch.stack((cage_src[:, 1], (1.0 - cage_src[:, 0])), dim=1)
        tex = TexturesUV(verts_uvs=verts_uvs[None], faces_uvs=face_vid[None], maps=texture_image)
        return tex

    @staticmethod
    def warp(anchor_deformed, faces, texture):
        verts = []
        for i in range(len(anchor_deformed)):
            cage_dst_npc = torch.stack((2.0 * anchor_deformed[i][:, 1] - 1.0, 1 - 2.0 * anchor_deformed[i][:, 0],), dim=1)
            cage_dst_npc = torch.cat((cage_dst_npc, -torch.ones(len(cage_dst_npc), 1).to(device)), dim=1).float()
            verts.append(cage_dst_npc)
        meshes = Meshes(verts=verts, faces=faces, textures=texture.extend(len(verts)))
        meshes = meshes.to(device)
        img_part_warp = ARAP_Rendering_Deformer.render_img(meshes)
        return img_part_warp

    @staticmethod
    def laplacian(verts, faces):
        dist_max = torch.sum((verts[None, ...] - verts[:,None,:])**2, dim=-1)
        lap = []
        for vid in range(len(verts)):
            rids = torch.where(faces==vid)[0]
            nnids = torch.unique(faces[rids].flatten())
            nnids = nnids[nnids!=vid]
            lap_v = dist_max[vid, nnids]
            lap.append(lap_v)
        return lap

    @staticmethod
    def calc_lap_diff(lap_pred, lap_gt):
        lap_diff = 0.0
        for i in range(len(lap_pred)):
            lap_diff += torch.abs(lap_pred[i] - lap_gt[i]).mean()
        return lap_diff

    @staticmethod
    def estimate_rotations(pts_0, pts_1, nbrs, wgts):
        tru_hood = (pts_0[nbrs] - pts_0[:, None, :]) * wgts[..., None]
        rot_hood = (pts_1[nbrs] - pts_1[:, None, :])
        rot_hood = rot_hood.permute((0, 2, 1))
        U, s, V0 = svd(torch.matmul(rot_hood, tru_hood))
        R0 = torch.matmul(U, V0.transpose(2, 1)).detach()
        # dets = torch.det(R0)
        # V = torch.stack((V0[...,0], V0[...,1]* dets[:, None]), dim=-1)
        # R = torch.matmul(U, V.transpose(2, 1))
        return R0

    @staticmethod
    def build_rhs(pts_0, pts_1, R, nbrs, wgts):
        R_avg = (R[nbrs] + R[:, None]) * 0.5
        tru_hood = (pts_0[..., None] - pts_0[nbrs].permute((0, 2, 1))) * wgts[:, None, :]
        rhs = torch.sum(torch.matmul(R_avg, tru_hood.permute((0, 2, 1))[..., None]).squeeze(), dim=1)  # P, 2
        return rhs

    @staticmethod
    def arap_loss(pts_0, pts_1, L, nbrs, wgts):
        R = ARAP_Rendering_Deformer.estimate_rotations(pts_0, pts_1, nbrs, wgts)
        #R = torch.where(R.isnan(), torch.zeros_like(R), R)
        b = ARAP_Rendering_Deformer.build_rhs(pts_0, pts_1, R, nbrs, wgts)
        #b = torch.where(b.isnan(), torch.zeros_like(b), b)
        diff = ((torch.matmul(L.float(), pts_1) - b) ** 2)
        #tmp = torch.where(tmp.isnan(), torch.zeros_like(tmp), tmp)
        return diff.mean()

    @staticmethod
    def render_img(mesh):
        R, T = look_at_view_transform()  # default dist=1, elev=0.0, azim=0.0,
        cameras = FoVOrthographicCameras(device=device, R=R, T=T)

        # Define the settings for rasterization and shading.
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SimpleShader()
        )
        images = renderer(mesh)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(images[0, ..., :3].cpu().numpy())
        # plt.axis("off")
        # plt.show()
        # return np.round(images[0, ..., :3].cpu().numpy()*255).astype(np.uint8)
        return images[..., :3]

    def run(self, lr=0.015, num_iter=100, w_reg=0.05):
        src_anchor_shift = []
        for i in range(len(self.anchor_shift_init)):
            for pid in range(len(self.anchor_shift_init[i])):
                src_anchor_shift.append(self.anchor_shift_init[i][pid].requires_grad_(True))

        self.optimizer = torch.optim.Adam([{'params': src_anchor_shift, 'lr': lr}],
                                          lr=0.1, betas=(0.9, 0.999), weight_decay=0.0)

        loss_last = 1e7
        gamma = 0.2
        schedule = []
        moving_meter = MovingAverage(10)
        for i in range(num_iter+1):
            if i in schedule:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * gamma
            #torch.autograd.set_detect_anomaly(True)
            self.optimizer.zero_grad()

            # warp
            img_part_warp = []
            mask_part_warp = []
            pts_deformed = [[] for _ in range(len(self.target_imgs))]
            for pid in range(len(self.src_anchor)):
                anchors_deformed, tri_deformed = [], []
                for img_i in range(len(self.target_imgs)):
                    anchor_deformed_p = self.src_anchor[pid] + src_anchor_shift[img_i*len(self.src_anchor)+pid]
                    anchors_deformed.append(anchor_deformed_p)
                    tri_deformed.append(self.faces[pid])
                    pts_deformed[img_i].append(anchor_deformed_p)
                img_part_warp_pid = ARAP_Rendering_Deformer.warp(anchors_deformed, tri_deformed, self.texture_img[pid])
                mask_part_warp_pid = ARAP_Rendering_Deformer.warp(anchors_deformed, tri_deformed, self.texture_mask[pid])
                img_part_warp.append(img_part_warp_pid.unsqueeze(dim=1))
                mask_part_warp.append(1 - mask_part_warp_pid.unsqueeze(dim=1))
            img_part_warp = torch.cat(img_part_warp, dim=1)
            mask_part_warp = torch.cat(mask_part_warp, dim=1)
            pts_deformed = [torch.cat(pts_deformed[i], dim=0) for i in range(len(pts_deformed))]
            pts_deformed = torch.cat(pts_deformed, dim=0)

            loss_rigid = ARAP_Rendering_Deformer.arap_loss(self.pts, pts_deformed, self.L, self._nbrs, self._wgts)
            # merge all parts
            loss_recon = 0.0
            recon_imgs = torch.ones_like(self.target_imgs)
            for img_i in range(len(self.anchor_shift_init)):
                for pid in range(len(self.anchor_shift_init[img_i])):
                    recon_imgs[img_i] = torch.where(mask_part_warp[img_i][pid] > 0.2, img_part_warp[img_i][pid], recon_imgs[img_i])

            # recon_imgs_np = np.round(recon_imgs.detach().cpu().numpy()*255).astype(np.uint8)
            # cv2.imshow("recon_img", np.concatenate(list(recon_imgs_np), axis=1))
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            loss_recon += F.mse_loss(recon_imgs, self.target_imgs)

            loss = loss_recon + w_reg * loss_rigid
            loss.backward(retain_graph=True)
            self.optimizer.step()

            loss_new = moving_meter.next(loss.item())
            if i % 20 == 0:
                print(f"iter {i}, loss_tar: {loss_recon} {loss_rigid}")
            if np.abs(loss_new - loss_last) < 1e-8:
                break
            loss_last = loss_new

        recon_imgs_np = np.round(recon_imgs.detach().to("cpu").numpy() * 255).astype(np.uint8)
        img_part_warp_np = img_part_warp.detach().to("cpu").numpy()
        mask_part_warp_np = mask_part_warp.detach().to("cpu").numpy()

        deformed_points = []
        deformed_faces = []
        for i in range(len(self.anchor_shift_init)):
            cumsum_p = 0
            deformed_points_i = []
            deformed_faces_i = []
            for p in range(len(self.anchor_shift_init[i])):
                deformed_points_i.append(src_anchor_shift[i*len(self.src_anchor)+p].detach().to("cpu").numpy() + self.src_anchor[p].to("cpu").numpy())
                deformed_faces_i.append(self.faces[p].to("cpu").numpy() + cumsum_p)
                cumsum_p += len(self.src_anchor[p])
            deformed_points.append(np.concatenate(deformed_points_i, axis=0))
            deformed_faces.append(np.concatenate(deformed_faces_i, axis=0))

        # for img_i in range(len(self.target_imgs)):
        #     tar_img = np.round(self.target_imgs[img_i].to("cpu").numpy() * 255).astype(np.uint8)
        #     cv2.imshow("recon_img", np.concatenate((recon_imgs_np[img_i], tar_img), axis=0))
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
        return recon_imgs_np, img_part_warp_np, mask_part_warp_np, deformed_points, deformed_faces


if __name__ == "__main__":
    from utils.vis_utils import visualize_cage, visualize_corr
    from inference.deform import create_cage
    from skimage import transform
    from skimage.transform import SimilarityTransform

    src_image = cv2.imread("/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test_sheet/yoga/yoga_1.png")
    src_mask_list = np.load("/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test_sheet/yoga/yoga_1_part_mask_list.npy")
    dst_image = cv2.imread("/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test_sheet/yoga/yoga_3.png")
    dst_mask = cv2.imread("/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test_sheet/yoga/yoga_3_mask.png", 0)
    corr = np.load("/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test_sheet/yoga/yoga_13_corr.npy")

    src_part_list = []
    for i in range(len(src_mask_list)):
        part_image = src_image * np.repeat(src_mask_list[i, ..., None]>0, 3, axis=-1) + np.ones_like(src_image)*255*np.repeat(src_mask_list[i, ..., None]==0, 3, axis=-1)
        # cv2.imshow("tar", np.concatenate((part_image, src_image), axis=1))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        src_part_list.append(part_image)
    src_part_list = np.stack(src_part_list, axis=0)

    cage_all, face_vid_all = [], []
    for pid in range(len(src_part_list)):
        _, cage_i = create_cage(src_part_list[pid], src_mask_list[pid], type="obb_grid")
        tri_vis = Delaunay(cage_i)
        face_vid = tri_vis.simplices
        #img_cage_src = visualize_cage(src_mask_list[pid], cage_i, tri_face=face_vid, show=True)
        #cage_all.append((cage_i / RES_DIM).astype(np.float32))
        cage_all.append(cage_i)
        face_vid_all.append(face_vid)

    src_anchor_shift = []
    for pid in range(len(src_part_list)):
        cage_pid = cage_all[pid]
        pts_src = np.argwhere(np.logical_and(src_mask_list[pid], np.all(corr > -1, axis=-1)))
        pts_tar = corr[pts_src[:, 0], pts_src[:, 1], 0:2]
        rtran = transform.estimate_transform('similarity', pts_src / RES_DIM, pts_tar / RES_DIM)
        cage_p_i = rtran(cage_pid)
        #visualize_cage(dst_mask, np.round(cage_p_i*255).astype(np.int32), face_vid_all[pid], show=True)
        src_anchor_shift.append((cage_p_i - cage_pid).astype(np.float32))

    deformer = ARAP_Rendering_Deformer(src_part_list / RES_DIM, src_mask_list / RES_DIM, dst_image[None, ...] / RES_DIM,
                                       face_vid_all, cage_all, [src_anchor_shift])
    recon_imgs, img_parts_warp, mask_parts_warp, mesh_V, mesh_F = deformer.run(lr=0.001, num_iter=100, w_reg=0.1)  # lr=0.001, num_iter=80, w_reg=0.05

    cv2.imshow("deformed_img", recon_imgs.squeeze(0))
    cv2.waitKey()
    cv2.destroyAllWindows()