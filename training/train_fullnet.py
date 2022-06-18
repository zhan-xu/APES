import sys
sys.path.append("./")
import argparse, os, cv2, shutil, numpy as np, time
import torch, torch.nn.functional as F, torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import models
from models.clusternet import sync_motion_seg
from models.losses import infoNCE, motionLoss, groupingLoss, iouLoss_slic
from datasets.okay_samurai_pair import OkaySamuraiPair
from utils.log_utils import AverageMeter
from utils.os_utils import isdir, mkdir_p, isfile
from utils.vis_utils import visualize_seg, visualize_corr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def slic_seg_to_pixel(slic, seg):
    """
    :param slic: H, W
    :param seg:  50, TRAIN_S
    """
    train_s = seg.shape[-1]
    slic_noneg = slic.clone()
    slic_noneg[slic_noneg == -1] = 0  # temporarily turn -1 to 0. They will be masked out
    pred_seg_pixel = torch.gather(seg, dim=0, index=slic_noneg.reshape(-1, 1).repeat(1, train_s))  # n_pix, self.train_s
    pred_seg_pixel = pred_seg_pixel.reshape(slic_noneg.shape[0], slic_noneg.shape[1], train_s)
    return pred_seg_pixel


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [0,1] x [0, 1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = x / (w - 1)
    y = y / (h - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([yy.unsqueeze_(0), xx.unsqueeze_(0)], 0)

    return meshed


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def get_losses(res, gt_corr, slic_label):
    if res["img1_feature"] is None:
        loss_corr = torch.zeros(1).to(device)
    else:
        loss_corr = infoNCE(res["img1_feature"], res["img2_feature"], gt_corr, res["tau"])
    slic_label_one_hot = F.one_hot(slic_label.long(), num_classes=-1).to(device)
    sim_mats_gt = torch.matmul(slic_label_one_hot.float(), slic_label_one_hot.transpose(2, 1).float())
    loss_motion = motionLoss(res["D"], sim_mats_gt)
    loss_group = groupingLoss(res["A"], sim_mats_gt)
    loss_iou = iouLoss_slic(res["S"], slic_label_one_hot)
    return {"corr": loss_corr, "motion": loss_motion, "group": loss_group, "iou": loss_iou}


def train(train_loader, model, optimizer):
    global device
    model.train()  # switch to train mode
    loss_corr_meter = AverageMeter()
    loss_motion_meter = AverageMeter()
    loss_group_meter = AverageMeter()
    loss_iou_meter = AverageMeter()
    loss_meter = AverageMeter()

    src_grid = make_coordinate_grid((256, 256), torch.float32)
    for data_pack in train_loader:
        img_1, img_2 = data_pack["img1"].to(device), data_pack["img2"].to(device)
        mask_1, mask_2 = data_pack["mask1"].float().to(device), data_pack["mask2"].float().to(device)
        gt_corr = data_pack["corr"].to(device)
        src_pixel_group = data_pack["src_pixel_group"].to(device)
        slic, slic_label = data_pack["slic"].to(device), data_pack["slic_label"].to(device)
        img_grid = src_grid[None, ...].repeat(img_1.shape[0], 1, 1, 1).to(device)
        pred_corr = data_pack["pred_corr"].to(device)

        optimizer.zero_grad()
        # forward
        res_f = model(img_1, img_2, mask_1, mask_2, img_grid, slic[..., 0], src_pixel_group[..., :2], pred_corr[..., :2])
        loss_f = get_losses(res_f, gt_corr[..., :2], slic_label[..., 0])
        # backward
        res_b = model(img_2, img_1, mask_2, mask_1, img_grid, slic[..., 1], src_pixel_group[..., 2:], pred_corr[..., 2:])
        loss_b = get_losses(res_b, gt_corr[..., 2:], slic_label[..., 1])

        loss_corr = (loss_f["corr"] + loss_b["corr"]) / 2.0
        loss_motion = (loss_f["motion"] + loss_b["motion"]) / 2.0
        loss_group = (loss_f["group"] + loss_b["group"]) / 2.0
        loss_iou = (loss_f["iou"] + loss_b["iou"]) / 2.0
        loss = loss_corr + loss_motion + loss_group + loss_iou
        loss.backward()
        optimizer.step()

        loss_corr_meter.update(loss_corr.item())
        loss_motion_meter.update(loss_motion.item())
        loss_group_meter.update(loss_group.item())
        loss_iou_meter.update(loss_iou.item())
        loss_meter.update(loss.item())
    return {"corr_loss": loss_corr_meter.avg, "motion_loss": loss_motion_meter.avg,
            "group_loss": loss_group_meter.avg, "iou_loss": loss_iou_meter.avg, "total_loss": loss_meter.avg}


def test(test_loader, model, save_result=False):
    global device
    model.eval()  # switch to test mode
    loss_corr_meter = AverageMeter()
    loss_motion_meter = AverageMeter()
    loss_group_meter = AverageMeter()
    loss_iou_meter = AverageMeter()
    loss_meter = AverageMeter()

    src_grid = make_coordinate_grid((256, 256), torch.float32)
    for data_pack in test_loader:
        img_1, img_2 = data_pack["img1"].to(device), data_pack["img2"].to(device)
        mask_1, mask_2 = data_pack["mask1"].float().to(device), data_pack["mask2"].float().to(device)
        seg_1, seg_2 = data_pack["seg1"].to(device), data_pack["seg2"].to(device)
        gt_corr = data_pack["corr"].to(device)
        src_pixel_group = data_pack["src_pixel_group"].to(device)
        slic, slic_label = data_pack["slic"].to(device), data_pack["slic_label"].to(device)
        img_grid = src_grid[None, ...].repeat(img_1.shape[0], 1, 1, 1).to(device)
        img1_filenames = data_pack["img1_filename"]
        pred_corr = data_pack["pred_corr"].to(device)

        with torch.no_grad():
            # forward
            res_f = model(img_1, img_2, mask_1, mask_2, img_grid, slic[..., 0], src_pixel_group[..., :2], pred_corr[..., :2])
            loss_f = get_losses(res_f, gt_corr[..., :2], slic_label[..., 0])
            # backward
            res_b = model(img_2, img_1, mask_2, mask_1, img_grid, slic[..., 1], src_pixel_group[..., 2:], pred_corr[..., 2:])
            loss_b = get_losses(res_b, gt_corr[..., 2:], slic_label[..., 1])

            loss_corr = (loss_f["corr"] + loss_b["corr"]) / 2.0
            loss_motion = (loss_f["motion"] + loss_b["motion"]) / 2.0
            loss_group = (loss_f["group"] + loss_b["group"]) / 2.0
            loss_iou = (loss_f["iou"] + loss_b["iou"]) / 2.0
            loss = loss_corr + loss_motion + loss_group + loss_iou

        loss_corr_meter.update(loss_corr.item())
        loss_motion_meter.update(loss_motion.item())
        loss_group_meter.update(loss_group.item())
        loss_iou_meter.update(loss_iou.item())
        loss_meter.update(loss.item())

        if save_result:
            outdir = "results/train_cluster/"
            if not os.path.exists(outdir):
                mkdir_p(outdir)
            pred_Af = torch.sigmoid(res_f["A"])
            pred_Ab = torch.sigmoid(res_b["A"])
            for i in range(len(img_1)):
                model_id = img1_filenames[i]
                print("processing: ", model_id)
                img_1_np = np.round(img_1[i].to("cpu").numpy().transpose(1, 2, 0)*255).astype(np.uint8)
                img_2_np = np.round(img_2[i].to("cpu").numpy().transpose(1, 2, 0)*255).astype(np.uint8)
                mask_1_np = mask_1[i].to("cpu").numpy().astype(np.uint8)
                mask_2_np = mask_2[i].to("cpu").numpy().astype(np.uint8)
                seg_gt_1_np = seg_1[i].to("cpu").numpy()
                seg_gt_2_np = seg_2[i].to("cpu").numpy()

                # seg
                # forward
                pred_Af_i = pred_Af[i]
                pred_Gf_i = sync_motion_seg(pred_Af_i[None, ...], cut_thres=0.01)
                pred_Gf_i = pred_Gf_i / (pred_Gf_i.sum(-1, keepdim=True) + 1e-6)
                seg_pred_1 = slic_seg_to_pixel(slic[i, :, :, 0].long(), pred_Gf_i.squeeze(0))
                seg_pred_1_np = torch.argmax(seg_pred_1, -1).to("cpu").numpy()
                seg_pred_1_np[mask_1_np == False] = -1
                seg_gt_1_np[mask_1_np == False] = -1

                img_seg_pred_1 = visualize_seg(img_1_np, seg_pred_1_np)
                img_seg_gt_1 = visualize_seg(img_1_np, seg_gt_1_np)
                img_show_1 = np.concatenate((img_seg_pred_1, img_seg_gt_1), axis=1)
                cv2.imshow("img_show", img_show_1)
                cv2.waitKey()
                cv2.destroyAllWindows()
                # cv2.imwrite(os.path.join(outdir, f"{model_id}_f.png"), img_show_1)
                # np.save(os.path.join(outdir, f"{model_id}_0_pred_seg.npy"), seg_pred_1_np)
                # np.save(os.path.join(outdir, f"{model_id}_0_gt_seg.npy"), seg_gt_1_np)
                # np.save(os.path.join(outdir, f"{model_id}_gt_corr.npy"), gt_corr_i)

                # backward
                pred_Ab_i = pred_Ab[i]
                pred_Gb_i = sync_motion_seg(pred_Ab_i[None, ...], cut_thres=0.01)
                pred_Gb_i = pred_Gb_i / (pred_Gb_i.sum(-1, keepdim=True) + 1e-6)
                seg_pred_2 = slic_seg_to_pixel(slic[i, :, :, 1].long(), pred_Gb_i.squeeze(0))
                seg_pred_2_np = torch.argmax(seg_pred_2, -1).to("cpu").numpy()
                seg_pred_2_np[mask_2_np == False] = -1
                seg_gt_2_np[mask_2_np == False] = -1

                img_seg_pred_2 = visualize_seg(img_2_np, seg_pred_2_np)
                img_seg_gt_2 = visualize_seg(img_2_np, seg_gt_2_np)
                img_show_2 = np.concatenate((img_seg_pred_2, img_seg_gt_2), axis=1)
                cv2.imshow("img_show", img_show_2)
                cv2.waitKey()
                cv2.destroyAllWindows()
                # cv2.imwrite(os.path.join(outdir, f"{model_id}_b.png"), img_show_2)
                # np.save(os.path.join(outdir, f"{model_id}_1_pred_seg.npy"), seg_pred_2_np)
                # np.save(os.path.join(outdir, f"{model_id}_1_gt_seg.npy"), seg_gt_2_np)
                # np.save(os.path.join(outdir, f"{model_id}_gt_corr.npy"), gt_corr_i)

    return {"corr_loss": loss_corr_meter.avg, "motion_loss": loss_motion_meter.avg,
            "group_loss": loss_group_meter.avg, "iou_loss": loss_iou_meter.avg, "total_loss": loss_meter.avg}


def main(args):
    global device
    lowest_loss = 1e20

    # create checkpoint dir and log dir
    if not isdir(args.checkpoint):
        print("Create new checkpoint folder " + args.checkpoint)
    mkdir_p(args.checkpoint)
    if not args.resume:
        if isdir(args.logdir):
            shutil.rmtree(args.logdir)
        mkdir_p(args.logdir)

    model = models.__dict__[args.arch](n_channels=4, n_classes=args.output_dim, bilinear=True, train_s=args.train_s, offline_corr=args.offline_corr)
    model.to(device)

    # load pretrained models
    model.corrnet.load_state_dict(torch.load(args.init_corrnet_path)['state_dict'])
    if args.offline_corr:
        for param in model.corrnet.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)

    else:
        optimizer = torch.optim.Adam([{'params': model.corrnet.parameters(), 'lr': args.lr_corrnet},
                                      {"params": model.clusternet.parameters()}],
                                     lr=args.lr, weight_decay=args.weight_decay)

    if args.init_fullnet_path:
        checkpoint = torch.load(args.init_fullnet_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("loaded fullnet from '{}'".format(args.init_fullnet_path))

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_loader = DataLoader(OkaySamuraiPair(root=args.train_folder, train_corr=False, color_jittering=True),
                              batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(OkaySamuraiPair(root=args.val_folder, train_corr=False),
                            batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(OkaySamuraiPair(root=args.test_folder, train_corr=False),
                             batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    if args.evaluate:
        print('\nEvaluation only')
        test_losses = test(test_loader, model, save_result=True)
        for loss_name, loss_value in test_losses.items():
            print(f"test_{loss_name}: {loss_value:6f}. ", end="")
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)
    logger = SummaryWriter(log_dir=args.logdir)
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("start time: ", current_time)
    for epoch in range(args.start_epoch, args.epochs):
        lr = scheduler.get_last_lr()
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr[0]))
        train_losses = train(train_loader, model, optimizer)
        val_losses = test(val_loader, model)
        test_losses = test(test_loader, model)
        scheduler.step()

        losses = [train_losses, val_losses, test_losses]
        for split_id, split_name in enumerate(["train", "val", "test"]):
            print(f"Epoch{epoch + 1}. ", end="")
            for loss_name, loss_value in losses[split_id].items():
                print(f"{split_name}_{loss_name}: {loss_value:6f}. ", end="")
                logger.add_scalar(f"{split_name}_{loss_name}", loss_value, epoch + 1)
            print("")
        # remember best acc and save checkpoint
        is_best = val_losses["group_loss"] < lowest_loss
        lowest_loss = min(val_losses["group_loss"], lowest_loss)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'lowest_loss': lowest_loss,
                         'optimizer': optimizer.state_dict()}, is_best, checkpoint=args.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full network')
    parser.add_argument('--arch', default='fullnet')
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)
    parser.add_argument('--lr_corrnet', default=1e-6, type=float)
    parser.add_argument('--schedule', type=int, nargs='+', default=[5])
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--workers', default=0, type=int)

    parser.add_argument('--train_batch', default=2, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--test_batch', default=2, type=int, metavar='N', help='test batchsize')
    parser.add_argument('-c', '--checkpoint', default='checkpoints/test', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--logdir', default='logs/test', type=str, metavar='LOG', help='directory to save logs')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--train_folder', default='/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/train/',
                        type=str, help='folder of training data')
    parser.add_argument('--val_folder', default='/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/val/',
                        type=str, help='folder of validation data')
    parser.add_argument('--test_folder', default='/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test/',
                        type=str, help='folder of testing data')
    parser.add_argument('--output_dim', default=64, type=int, metavar='N', help='output_dim')
    parser.add_argument('--train_s', default=12, type=int, metavar='N', help='training segmentation number')
    parser.add_argument('--init_corrnet_path', default="../checkpoints/pretrain_corrnet_os/model_best.pth.tar", type=str)
    parser.add_argument('--init_fullnet_path', default="", type=str) #../checkpoints/train_cluster/model_best.pth.tar
    parser.add_argument('--offline_corr', action='store_true')

    print(parser.parse_args())
    main(parser.parse_args())
