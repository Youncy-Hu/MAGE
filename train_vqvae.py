import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid

from modules.vqvae_model import VectorQuantizedVAE
from dataload import MNIST4VQVAE, CATER4VQVAE

from tensorboardX import SummaryWriter
from collections import OrderedDict
import torch.distributed as dist

def train(data_loader, model, optimizer, args, writer):
    for images in data_loader:
        images = images.to(args.device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + args.beta * loss_commit
        loss.backward()

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)

        optimizer.step()
        args.steps += 1

def test(data_loader, model, args, writer, ngpus_per_node):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images in data_loader:
            images = images.to(args.device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    if args.multiprocessing_distributed:
        dist.barrier()
        dist.all_reduce(loss_recons, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_vq, op=dist.ReduceOp.SUM)
        loss_recons /= dist.get_world_size()
        loss_vq /= dist.get_world_size()
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                               and args.rank % ngpus_per_node == 0):
        writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/test/quantization', loss_vq.item(), args.steps)

    return loss_recons.item(), loss_vq.item()

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde

def main(gpu, ngpus_per_node, args):
    writer = SummaryWriter('{0}/{1}'.format(args.log_folder, args.output_folder))
    save_filename = '{0}/{1}'.format(args.model_folder, args.output_folder)

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[1.0]),
        ])
        train_dataset = MNIST4VQVAE(args.data_root, 'train', image_transform=transform)
        test_dataset = MNIST4VQVAE(args.data_root, 'test', image_transform=transform)
        num_channels = 1
        down_ratio = 4
    elif args.dataset == 'cater_gen':
        num_channels = 3
        down_ratio = 8
        transform = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.8, 1.)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        train_dataset = CATER4VQVAE(args.data_root, 'train', image_transform=transform)
        test_dataset = CATER4VQVAE(args.data_root, 'test', image_transform=transform)

    model = VectorQuantizedVAE(num_channels, down_ratio, args.hidden_size, args.k)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    name = k[7:]
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=16, shuffle=False, drop_last=False, sampler=test_sampler)

    # Fixed images for Tensorboard
    fixed_images = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, args)
    grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('reconstruction', grid, 0)
    writer.add_text('data', f"```\n{args}\n```")

    best_loss = -1.
    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, args, writer)
        loss, _ = test(test_loader, model, args, writer, ngpus_per_node)

        reconstruction = generate_samples(fixed_images, model, args)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        writer.add_image('reconstruction', grid, epoch + 1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                   and args.rank % ngpus_per_node == 0):
            print("epoch %d, test_loss = %.6f" % (epoch, loss))
            if (epoch == 0) or (loss < best_loss):
                best_loss = loss
                with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                    torch.save(model.state_dict(), f)
            with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
                torch.save(model.state_dict(), f)

if __name__ == '__main__':
    import argparse
    import os
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-root', type=str, default='./data/moving_mnist/mnist_single_20f_10k_',
        help='name of the data folder')
    parser.add_argument('--dataset', type=str, default='mnist', choices=["mnist", "cater-gen"])

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=2.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='mnist_512_256',
        help='name of the output folder')
    parser.add_argument('--log-folder', type=str, default='./models/log')
    parser.add_argument('--model-folder', type=str, default='./models/model')
    parser.add_argument('--num-workers', type=int, default=4,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda, default: cpu)')

    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:65532', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true')
    parser.add_argument("--resume", type=str, default='')

    args = parser.parse_args()

    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    if not os.path.exists('{0}/{1}'.format(args.model_folder, args.output_folder)):
        os.makedirs('{0}/{1}'.format(args.model_folder, args.output_folder))
    args.steps = 0

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main(args.gpu, ngpus_per_node, args)
