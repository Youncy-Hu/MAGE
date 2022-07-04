import argparse
import os
import sys
sys.path.append(os.getcwd())
from torch.utils.data import DataLoader
from dataload import MovingMnistLMDB, CaterGEN
from modules.vqvae_model import VectorQuantizedVAE
from modules.MAGE_model import MAGE
import matplotlib.pyplot as plt
from utils.videotransforms import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cater-gen-v1', choices=["mnist", "cater-gen-v1", "cater-gen-v2"])
parser.add_argument('--data-root', type=str, default='../datasets/CATER-GEN-v1')
parser.add_argument('--vqvae-model', type=str, default='./models/vqvae-cater-gen-v1-512-256.pt')
parser.add_argument('--mage-model', type=str, default='./models/cater-gen-v1/iteration_33000.pth')
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

parser.add_argument('--encode-dim', type=int, default=256)
parser.add_argument('--codebook-size', type=int, default=512)
parser.add_argument('--frames-length', type=int, default=10)
parser.add_argument('--resolution', type=int, default=16)
parser.add_argument('--speed', type=int, default=[3.0, 6.0])
parser.add_argument('--vision-width', type=int, default=512)
parser.add_argument('--dropout-rate', type=int, default=0.1)
parser.add_argument('--text-length', type=int, default=32)
parser.add_argument('--vocab-size', type=int, default=30)
parser.add_argument('--transformer-width', type=int, default=512)
parser.add_argument('--transformer-heads', type=int, default=16)
parser.add_argument('--transformer-layers', type=int, default=2)
parser.add_argument('--randomness', type=bool, default=False)
# fmt: on

def test(opt):

    if opt.dataset == 'mnist':
        opt.speed = [1.0, 2.0]
        test_dataset = MovingMnistLMDB(opt.data_root, 'test', opt.frames_length, opt.speed, image_transform=None)
        channel_dim = 1
        down_ratio = 4
        opt.text_length = 20
    elif opt.dataset == 'cater-gen-v1':
        transform = Compose([
            Resize(128),
            ClipToTensor(channel_nb=3),
            Normalize(mean=[0.5], std=[0.5])
        ])
        test_dataset = CaterGEN(opt.dataset, opt.data_root, 'test', opt.frames_length, opt.speed, image_transform=transform, randomness=opt.randomness)
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
        test_dataset = CaterGEN(opt.dataset, opt.data_root, 'test', opt.frames_length, opt.speed, image_transform=transform, randomness=opt.randomness)
        channel_dim = 3
        down_ratio = 8
        opt.text_length = 38
        opt.vocab_size = 50
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    vqvae = VectorQuantizedVAE(input_dim=channel_dim, down_ratio=down_ratio, dim=opt.encode_dim, K=opt.codebook_size).to(opt.device)
    with open(opt.vqvae_model, 'rb') as f:
        state_dict = torch.load(f)
        vqvae.load_state_dict(state_dict)
    vqvae.eval()

    opt.dropout_rate = 0.
    model = MAGE(
        opt.codebook_size,
        opt.frames_length,
        opt.resolution,
        opt.vision_width,
        opt.text_length,
        opt.vocab_size,
        opt.transformer_width,
        opt.transformer_heads,
        opt.transformer_layers,
        opt.dropout_rate,
        opt.randomness
    )
    model = model.to(opt.device)

    if os.path.isfile(opt.mage_model):
        if opt.gpu is None:
            checkpoint = torch.load(opt.mage_model)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(opt.gpu)
            checkpoint = torch.load(opt.mage_model, map_location=loc)
            # new_state_dict = OrderedDict()
            # for k, v in checkpoint['state_dict'].items():
            #     name = k[7:]  # discard ``module."
            #     new_state_dict[name] = v
            # model.load_state_dict(new_state_dict)
            model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.mage_model, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(opt.mage_model))
    model.eval()

    with torch.no_grad():
        idx = 0
        for images, captions, speed, caption_len in test_dataloader:
            seqs = images.to(opt.device)
            caption_tokens = captions.to(opt.device)
            speed = speed.to(opt.device)
            caption_len = caption_len.to(opt.device)

            latents = vqvae.encode(seqs.view(-1, *images.size()[2:]))
            latents = latents.view(*images.size()[:2], *latents.size()[1:])
            latents = latents.detach()
            vq_imgs = vqvae.decode(latents=latents[0]).detach()

            for i in range(opt.frames_length - 1):
                if i == 0:
                    imgs = torch.zeros_like(latents)
                    imgs[:, 0, :, :] = latents[:, 0, :, :]
                else:
                    imgs[:, i, :, :] = prediction_id[:, i-1, :, :]

                if opt.randomness:
                    prediction, loss2, loss3 = model(imgs, caption_tokens, speed, caption_len, test_flag=True)
                else:
                    prediction = model(imgs, caption_tokens, speed, caption_len)
                prediction_id = torch.max(prediction, -1)[1]
                prediction_id = prediction_id.view(imgs.size()[0], opt.frames_length-1, *imgs.size()[2:])

            imgs[:, opt.frames_length-1, :, :] = prediction_id[:, -1, :, :]
            prediction_imgs = vqvae.decode(latents=imgs[0]).detach()

            show_gif(vq_imgs[:opt.frames_length].cpu(), prediction_imgs[:opt.frames_length].cpu(), test_dataset.tokens2sent(caption_tokens[0].cpu().numpy()), speed.cpu().numpy(), channel_dim, opt)
            print(idx)
            idx += 1

def show_gif(src_imgs, tgr_imgs, caption, speed, channel_dim, opt):
    src_imgs.clamp_(min=-1, max=1)
    tgr_imgs.clamp_(min=-1, max=1)
    src_imgs.add_(1).div_(2 + 1e-5)
    tgr_imgs.add_(1).div_(2 + 1e-5)
    src_imgs = (src_imgs * 255.).numpy()
    tgr_imgs = (tgr_imgs * 255.).numpy()
    for i in range(src_imgs.shape[0]):
        plt.subplot(2, opt.frames_length, i + 1)
        if channel_dim == 1:
            plt.imshow(src_imgs[i, 0, :, :].astype(np.uint8), cmap='gray')
        else:
            plt.imshow(src_imgs[i, :, :, :].astype(np.uint8).transpose(1, 2, 0))
        plt.axis('off')
    for j in range(tgr_imgs.shape[0]):
        plt.subplot(2, opt.frames_length, j + opt.frames_length + 1)
        if channel_dim == 1:
            plt.imshow(tgr_imgs[j, 0, :, :].astype(np.uint8), cmap='gray')
        else:
            plt.imshow(tgr_imgs[j, :, :, :].astype(np.uint8).transpose(1, 2, 0))
        plt.axis('off')
    plt.suptitle(caption + '& speed: ' + str(speed))
    plt.show()

if __name__ == "__main__":
    opt = parser.parse_args()
    test(opt)
