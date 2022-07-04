import argparse
import os
import sys
sys.path.append(os.getcwd())
import shutil
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
from modules.vqvae_model import VectorQuantizedVAE
import torch.distributed as dist
from utils.timer import Timer
import torch.optim as optim
from dataload import MovingMnistLMDB, CaterGEN
from modules.MAGE_model import MAGE
from utils.videotransforms import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cater-gen-v1', choices=["mnist", "cater-gen-v1", "cater-gen-v2"])
parser.add_argument('--data-root', type=str, default='../datasets/CATER-GEN-v1')
parser.add_argument('--vqvae-model', type=str, default='./models/vqvae-cater-gen-v1-512-256.pt')
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--epoch', type=int, default=151)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--lr-gamma', type=float, default=0.1)
parser.add_argument('--lr-steps', type=list, default=[30, 40])
parser.add_argument('--cos', default=True, type=bool, help='use cosine lr schedule')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument('--checkpoint-path', type=str, default='./models/cater-gen-v1/')
parser.add_argument("--checkpoint-every", type=int, default=500, help='save the checkpoint')
parser.add_argument("--log-every", type=int, default=1)
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

parser.add_argument('--encode-dim', type=int, default=256, help='consistent with VQVAE')
parser.add_argument('--codebook-size', type=int, default=512, help='consistent with VQVAE')
parser.add_argument('--frames-length', type=int, default=10)
parser.add_argument('--speed', type=int, default=[3.0, 6.0], help='range of sampling interval')
parser.add_argument('--resolution', type=int, default=16)
parser.add_argument('--vision-width', type=int, default=512)
parser.add_argument('--dropout-rate', type=int, default=0.1)
parser.add_argument('--text-length', type=int, default=32)
parser.add_argument('--vocab-size', type=int, default=30)
parser.add_argument('--transformer-width', type=int, default=512)
parser.add_argument('--transformer-heads', type=int, default=16)
parser.add_argument('--transformer-layers', type=int, default=2)
parser.add_argument('--randomness', type=bool, default=False, help='False for deterministic video generation, True for stochastic video generation')
parser.add_argument('--alpha', type=float, default=0.0001, help='Only used when randomness=True')
parser.add_argument('--beta', type=float, default=0.0005, help='Only used when randomness=True')
# fmt: on

def train(gpu, ngpus_per_node, opt):
    opt.gpu = gpu

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            opt.rank = opt.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

    if opt.dataset == 'mnist':
        opt.speed = [1.0, 2.0]
        train_dataset = MovingMnistLMDB(opt.data_root, 'train', opt.frames_length, opt.speed, image_transform=None)
        channel_dim = 1
        down_ratio = 4
        opt.text_length = 20
    elif opt.dataset == 'cater-gen-v1':
        transform = Compose([
            Resize(128),
            ClipToTensor(channel_nb=3),
            Normalize(mean=[0.5], std=[0.5])
        ])
        train_dataset = CaterGEN(opt.dataset, opt.data_root, 'train', opt.frames_length, opt.speed, image_transform=transform, randomness=opt.randomness)
        channel_dim = 3
        down_ratio = 8
        opt.text_length = 32
        opt.vocab_size = 30
    elif opt.dataset == 'cater-gen-v2':
        transform = Compose([
            Resize(128),
            ClipToTensor(channel_nb=3),
            Normalize(mean=[0.5], std=[0.5])
        ])
        train_dataset = CaterGEN(opt.dataset, opt.data_root, 'train', opt.frames_length, opt.speed, image_transform=transform, randomness=opt.randomness)
        channel_dim = 3
        down_ratio = 8
        opt.text_length = 38
        opt.vocab_size = 50

    vqvae = VectorQuantizedVAE(input_dim=channel_dim, down_ratio=down_ratio, dim=opt.encode_dim, K=opt.codebook_size)
    with open(opt.vqvae_model, 'rb') as f:
        state_dict = torch.load(f)
        vqvae.load_state_dict(state_dict)

    model = MAGE(
        codebook_size=opt.codebook_size, frames_length=opt.frames_length, resolution=opt.resolution,
        vision_width=opt.vision_width, context_length=opt.text_length, vocab_size=opt.vocab_size,
        transformer_width=opt.transformer_width, transformer_heads=opt.transformer_heads,
        transformer_layers=opt.transformer_layers, dropout=opt.dropout_rate, randomness=opt.randomness
    )
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
            opt.batchsize = int(opt.batchsize / ngpus_per_node)
            opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
            vqvae.cuda(opt.gpu)
            vqvae = torch.nn.parallel.DistributedDataParallel(vqvae, device_ids=[opt.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            vqvae.cuda()
            vqvae = torch.nn.parallel.DistributedDataParallel(vqvae)
    elif opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        model = model.cuda(opt.gpu)
        vqvae = vqvae.cuda(opt.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        vqvae = torch.nn.DataParallel(vqvae).cuda()

    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=(train_sampler is None), drop_last=False,
                                  pin_memory=False, collate_fn=train_dataset.collate_fn, sampler=train_sampler,
                                  num_workers=opt.num_workers, persistent_workers=True)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9,0.98), eps=1e-6)
    start_epoch = 0
    cudnn.benchmark = True

    if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0):
        tensorboard_writer = SummaryWriter(log_dir=opt.checkpoint_path)
        tensorboard_writer.add_text('data', f"```\n{opt}\n```")
    timer = Timer(
        start_from=start_epoch, total_iterations=opt.epoch
    )
    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    model.train()
    vqvae.eval()
    iteration = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, opt.epoch):
        if opt.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, opt)

        for images, captions, speed, caption_len in train_dataloader:
            timer.tic()
            seqs = images.to(opt.device)
            caption_tokens = captions.to(opt.device)
            speed = speed.to(opt.device)
            caption_len = caption_len.to(opt.device)

            with torch.no_grad():
                seqs = seqs.view(-1, *images.size()[2:])
                latents = vqvae.encode(seqs)
                latents = latents.view(*images.size()[:2], *latents.size()[1:])
                latents = latents.detach()

            optimizer.zero_grad()
            if opt.randomness:
                prediction, kl_loss, l2_loss = model(latents, caption_tokens, speed, caption_len)
                loss = F.cross_entropy(prediction.view(-1, opt.codebook_size), latents[:, 1:opt.frames_length, :, :].contiguous().view(
                    -1)) + opt.beta * kl_loss + opt.alpha * l2_loss
            else:
                prediction = model(latents, caption_tokens, speed, caption_len)
                loss = F.cross_entropy(prediction.view(-1, opt.codebook_size), latents[:, 1:opt.frames_length, :, :].contiguous().view(-1))

            loss.backward()
            optimizer.step()
            timer.toc()

            iteration += 1
            if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0):
                print("iter %d (epoch %d), train_loss = %.6f" % (iteration, epoch, loss))
                tensorboard_writer.add_scalar('loss/train/prediction', loss.clone().detach().item(), iteration)
                tensorboard_writer.add_scalar('loss/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                if opt.randomness:
                    tensorboard_writer.add_scalar('loss/train/KL', kl_loss.clone().detach().item(), iteration)
                    tensorboard_writer.add_scalar('loss/train/regularize', l2_loss.clone().detach().item(), iteration)

            if iteration % opt.checkpoint_every == 0:
                if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed
                                                           and opt.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join(opt.checkpoint_path, 'iteration_%d.pth' % (iteration)))

def save_checkpoint(state, filename='work_dirs/checkpoint.pth', is_best=False):
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'model_best.pth'))

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
