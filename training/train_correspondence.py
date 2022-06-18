import sys
sys.path.append("./")
import argparse, os, shutil, numpy as np, time
import torch, torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import models
from models.losses import infoNCE
from datasets.okay_samurai_pair import OkaySamuraiPair
from datasets.creative_flow_pair import CreativeFlowPair
from utils.log_utils import AverageMeter
from utils.os_utils import isdir, mkdir_p, isfile
from utils.vis_utils import visualize_corr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def train(train_loader, model, optimizer):
    global device
    model.train()  # switch to train mode
    loss_meter = AverageMeter()

    for data_pack in train_loader:
        img_1, img_2, mask_1, mask_2, gt_corr = \
            data_pack["img1"].to(device), data_pack["img2"].to(device), \
            data_pack["mask1"].float().to(device), data_pack["mask2"].float().to(device), data_pack["corr"].to(device)
        optimizer.zero_grad()
        img1_feature, img2_feature, tau = model(img_1, img_2, mask_1.unsqueeze(1), mask_2.unsqueeze(1))
        loss_corr_forward = infoNCE(img1_feature, img2_feature, gt_corr[..., :2], tau)
        loss_corr_backward = infoNCE(img2_feature, img1_feature, gt_corr[..., 2:], tau)
        loss = (loss_corr_forward + loss_corr_backward) / 2.0
        loss_meter.update(loss.item())
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
    return {"total_loss": loss_meter.avg}


def test(test_loader, model, save_result=False):
    global device
    model.eval()  # switch to test mode
    loss_meter = AverageMeter()

    for data_pack in test_loader:
        img_1, img_2, mask_1, mask_2, gt_corr, img_names = \
            data_pack["img1"].to(device), data_pack["img2"].to(device), \
            data_pack["mask1"].float().to(device), data_pack["mask2"].float().to(device), \
            data_pack["corr"].to(device), data_pack["img1_filename"]
        with torch.no_grad():
            img1_feature, img2_feature, tau = model(img_1, img_2, mask_1.unsqueeze(1), mask_2.unsqueeze(1))
            loss_corr_forward = infoNCE(img1_feature, img2_feature, gt_corr[..., :2], tau)
            loss_corr_backward = infoNCE(img2_feature, img1_feature, gt_corr[..., 2:], tau)
            loss = (loss_corr_forward + loss_corr_backward) / 2.0
        loss_meter.update(loss.item())

        if save_result:
            #outdir = "results/corrnet_creative_flow"
            outdir = "results/corrnet_okaysamurai"
            if not os.path.exists(outdir):
               mkdir_p(outdir)
            for i in range(len(img_1)):
                img_name = img_names[i]
                print("processing: ", img_name)
                img1_feature_np = img1_feature[i].to("cpu").numpy().transpose(1, 2, 0)
                img2_feature_np = img2_feature[i].to("cpu").numpy().transpose(1, 2, 0)
                img_1_np = img_1[i].to("cpu").numpy().transpose(1, 2, 0)
                img_2_np = img_2[i].to("cpu").numpy().transpose(1, 2, 0)
                img_1_np = np.round(img_1_np * 255.0).astype(np.uint8)
                img_2_np = np.round(img_2_np * 255.0).astype(np.uint8)
                mask_1_np = (data_pack["mask1"][i] * 255).to("cpu").numpy()
                mask_2_np = (data_pack["mask2"][i] * 255).to("cpu").numpy()
                corr_np = gt_corr[i].to("cpu").numpy()

                # forward
                p_from = np.argwhere(np.all(corr_np[:, :, :2] > -1, axis=-1))
                p_tar_all = np.argwhere(mask_2_np)
                pf_from = img1_feature_np[p_from[:, 0], p_from[:, 1]]
                pf_tar_all = img2_feature_np[p_tar_all[:, 0], p_tar_all[:, 1]]
                similarity = np.matmul(pf_from, pf_tar_all.T) / tau.item()
                pairwise_nnind = np.argmax(similarity, axis=1)
                pt_match_pred = p_tar_all[pairwise_nnind]
                corr_pred_forward = -np.ones((corr_np.shape[0], corr_np.shape[1], 2)).astype(np.int64)
                corr_pred_forward[p_from[:, 0], p_from[:, 1]] = pt_match_pred
                img_forward = visualize_corr(img_1_np, np.all(corr_np[:, :, :2] > -1, axis=-1)*255, img_2_np, corr_pred_forward, show=True)

                # backward
                p_from = np.argwhere(np.all(corr_np[:, :, 2:] > -1, axis=-1))
                p_tar_all = np.argwhere(mask_1_np)
                pf_from = img2_feature_np[p_from[:, 0], p_from[:, 1]]
                pf_tar_all = img1_feature_np[p_tar_all[:, 0], p_tar_all[:, 1]]
                similarity = np.matmul(pf_from, pf_tar_all.T) / tau.item()
                pairwise_nnind = np.argmax(similarity, axis=1)
                pt_match_pred = p_tar_all[pairwise_nnind]
                corr_pred_backward = -np.ones((corr_np.shape[0], corr_np.shape[1], 2)).astype(np.int64)
                corr_pred_backward[p_from[:, 0], p_from[:, 1]] = pt_match_pred
                img_backward = visualize_corr(img_2_np, np.all(corr_np[:, :, 2:] > -1, axis=-1)*255, img_1_np, corr_pred_backward, show=True)

                model_id = img_name.replace(".png", "")
                np.save(os.path.join(outdir, f"{model_id}_corr_pred.npy"), np.concatenate((corr_pred_forward, corr_pred_backward), axis=-1))
                np.save(os.path.join(outdir, f"{model_id}_corr_gt.npy"), corr_np)

    return {"total_loss": loss_meter.avg}


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

    model = models.__dict__[args.arch](n_channels=4, n_classes=args.output_dim, bilinear=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)
    model.to(device)

    if args.init_corrnet_path:
        if isfile(args.init_corrnet_path):
            print("=> loading init weights '{}'".format(args.init_corrnet_path))
            checkpoint = torch.load(args.init_corrnet_path)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded init weights '{}' (epoch {})".format(args.init_corrnet_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.init_corrnet_path))
            exit()

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    if args.dataset == "okaysamurai":
        train_loader = DataLoader(OkaySamuraiPair(root=args.train_folder, train_corr=True, color_jittering=True),
                                  batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        val_loader = DataLoader(OkaySamuraiPair(root=args.val_folder, train_corr=True),
                                batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        test_loader = DataLoader(OkaySamuraiPair(root=args.test_folder, train_corr=True),
                                 batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    elif args.dataset == "creativeflow":
        train_loader = DataLoader(CreativeFlowPair(root=args.train_folder, color_jittering=True),
                                  batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        val_loader = DataLoader(CreativeFlowPair(root=args.val_folder), batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        test_loader = DataLoader(CreativeFlowPair(root=args.test_folder), batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    else:
        raise NotImplementedError

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
        is_best = val_losses["total_loss"] < lowest_loss
        lowest_loss = min(val_losses["total_loss"], lowest_loss)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'lowest_loss': lowest_loss,
                         'optimizer': optimizer.state_dict()}, is_best, checkpoint=args.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full network')
    parser.add_argument('--arch', default='corrnet')
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--schedule', type=int, nargs='+', default=[])
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--train_batch', default=2, type=int, help='train batchsize')
    parser.add_argument('--test_batch', default=2, type=int, help='test batchsize')

    parser.add_argument('-c', '--checkpoint', default='checkpoints/test', type=str, help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--logdir', default='logs/test', type=str, help='directory to save logs')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--dataset', default='okaysamurai', type=str)  # okaysamurai, creativeflow
    parser.add_argument('--train_folder', default='/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/train/', #creative_flow, OkaySamural_syn/train/
                        type=str, help='folder of training data')
    parser.add_argument('--val_folder', default='/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/val/',
                        type=str, help='folder of validation data')
    parser.add_argument('--test_folder', default='/mnt/neghvar/mnt/DATA_LINUX/zhan/puppets/OkaySamural_syn/test/',
                        type=str, help='folder of testing data')
    parser.add_argument('--init_corrnet_path', default='', type=str)  #../checkpoints/pretrain_corrnet_cf/model_best.pth.tar
    parser.add_argument('--output_dim', default=64, type=int, help='output_dim')

    print(parser.parse_args())
    main(parser.parse_args())
