import argparse
import os
import random
import sys
import cv2
os.environ['MKL_THREADING_LAYER'] = 'GNU'
sys.path.append(os.getcwd())
import shutil
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math
import numpy as np


import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
from utils.timer import Timer
import json
from omegaconf import OmegaConf
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import OrderedDict
from utils.util import instantiate_from_config

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/mage+_caterv1.yaml')
# parser.add_argument('--bert-path', type=str, default='../bert-base-uncased/')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--checkpoint-path', type=str, default='../results//')

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:65532', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument("--n_samples", type=int, default=1, help="how many samples to produce for each instance",)
parser.add_argument("--test_model", type=str, default='./models/MAGE+/catergenv2_deterministic/model_best.pth')

def train(gpu, ngpus_per_node, opt):
    opt.gpu = gpu

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    configs = OmegaConf.load(opt.config)
    if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0):
        os.makedirs(opt.checkpoint_path, exist_ok=True)
        OmegaConf.save(configs, os.path.join(opt.checkpoint_path, "config.yaml"))

    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.rank = opt.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

    train_dataset = instantiate_from_config(configs.data, {'split': 'train'})
    test_dataset = instantiate_from_config(configs.data, {'split': 'test'})  # validation for sthv2, val for kinetics
    model = instantiate_from_config(configs.model)

    if opt.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.gpu is not None:
            torch.cuda.set_device(opt.gpu)
            model.cuda(opt.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            configs.train.batchsize = int(configs.train.batchsize / ngpus_per_node)
            opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        model = model.cuda(opt.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
    train_dataloader = DataLoader(train_dataset, batch_size=configs.train.batchsize, shuffle=(train_sampler is None), drop_last=False,
                                  pin_memory=False, collate_fn=train_dataset.collate_fn if callable(getattr(train_dataset, 'collate_fn', None)) else None,
                                  sampler=train_sampler, num_workers=opt.num_workers, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.train.batchsize, shuffle=False, drop_last=False,
                                 pin_memory=False, collate_fn=test_dataset.collate_fn if callable(getattr(train_dataset, 'collate_fn', None)) else None,
                                 sampler=test_sampler, num_workers=opt.num_workers, persistent_workers=True)

    optimizer = optim.Adam(model.parameters(), lr=configs.train.lr, betas=(0.9,0.98), eps=1e-6)
    start_epoch = 0
    cudnn.benchmark = True

    if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0):
        tensorboard_writer = SummaryWriter(log_dir=opt.checkpoint_path)
        tensorboard_writer.add_text('data', f"```\n{opt}\n```")
    timer = Timer(
        start_from=start_epoch, total_iterations=configs.train.epoch
    )
    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    best_loss = 100.
    iteration = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, configs.train.epoch):
        if opt.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, configs.train)

        for batch in train_dataloader:
            if 'video_id' in batch.keys():
                del batch['video_id']
            for k in batch.keys():
                batch[k] = batch[k].to(opt.device)

            timer.tic()
            optimizer.zero_grad()
            loss, loss_dict = model(batch)

            loss.backward()
            optimizer.step()
            timer.toc()

            iteration += 1

            if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0):
                print("iter %d (epoch %d), train_loss = %.6f" % (iteration, epoch, loss.detach().item()))
                tensorboard_writer.add_scalars('loss/', loss_dict, iteration)
                tensorboard_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], iteration)

            if iteration % configs.train.checkpoint_every == 0:
                test_loss = 0.
                count = 0

                with torch.no_grad():
                    for batch in test_dataloader:
                        if 'video_id' in batch.keys():
                            del batch['video_id']
                        for k in batch.keys():
                            batch[k] = batch[k].to(opt.device)

                        loss, _ = model(batch)
                        test_loss += loss.detach()
                        count += 1
                    test_loss /= count

                if opt.multiprocessing_distributed:
                    dist.barrier()
                    dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
                    test_loss /= dist.get_world_size()
                is_best = test_loss.item() < best_loss
                best_loss = min(test_loss.item(), best_loss)
                if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed
                                                           and opt.rank % ngpus_per_node == 0):
                    print("iteration %d (epoch %d), test_loss = %.6f" % (iteration, epoch, test_loss))
                    tensorboard_writer.add_scalar('loss/val/final_loss', test_loss.item(), iteration)
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, is_best, os.path.join(opt.checkpoint_path, 'iteration_%d.pth' % (iteration)))

def save_checkpoint(state, is_best, filename='work_dirs/checkpoint.pth'):
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if is_best:
        torch.save(state, os.path.join(os.path.dirname(filename), 'model_best.pth'))

def sampling(opt):
    test_model = opt.test_model
    configs = OmegaConf.load(os.path.join(os.path.dirname(test_model), "config.yaml"))
    test_dataset = instantiate_from_config(configs.data, {'split': 'test'})
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    model = instantiate_from_config(configs.model)
    model = model.to(opt.device)

    if os.path.isfile(test_model):
        if opt.gpu is None:
            checkpoint = torch.load(test_model)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(opt.gpu)
            checkpoint = torch.load(test_model, map_location=loc)
            if list(checkpoint["state_dict"].keys())[0].startswith('module.'):
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(test_model))
    else:
        print("=> no checkpoint found at '{}'".format(test_model))

    model.eval()
    with torch.no_grad():
        idx = 0
        for batch in test_dataloader:
            if 'video_id' in batch.keys():
                video_id = batch['video_id'][0]
                del batch['video_id']
            for k in batch.keys():
                batch[k] = batch[k].to(opt.device)

            for iiidx in range(opt.n_samples):
                generated = model.autoregressive_generate(batch)
                generated.clamp_(min=-1, max=1)

            # save_name = video_id + '-' + '{:.4f}'.format(batch['speed'][0].cpu().numpy())
            # save_gifs(generated[0].cpu(), save_name, test_model)

            print(idx)
            idx += 1

def save_gifs(tgr, video_id, test_model):
    import imageio
    tgr_imgs = (tgr + 1) * 0.5
    tgr_imgs = (tgr_imgs * 255.).numpy().astype(np.uint8).transpose(0, 2, 3, 1)
    save_path = os.path.join(os.path.dirname(test_model), 'videos')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imageio.mimsave(os.path.join(save_path, video_id+'.gif'), tgr_imgs, fps=3)

def last_zeros(a):
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges[-1, 0]

def adjust_learning_rate(optimizer, epoch, opt):
    """Decay the learning rate based on schedule"""
    lr = opt.lr
    if opt.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / opt.epoch))
    else:  # stepwise lr schedule
        for milestone in opt.lr_steps:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    opt = parser.parse_args()

    if opt.split == 'train':
        if opt.dist_url == "env://" and opt.world_size == -1:
            opt.world_size = int(os.environ["WORLD_SIZE"])

        opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed

        ngpus_per_node = torch.cuda.device_count()
        if opt.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            opt.world_size = ngpus_per_node * opt.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
        else:
            # Simply call main_worker function
            train(opt.gpu, ngpus_per_node, opt)
    elif opt.split == 'test':
        sampling(opt)
