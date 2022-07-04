import torch
import torch.nn as nn

from torch.autograd import Function

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return codes, indices_flatten

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return grad_inputs, grad_codebook

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hid = dim_out // 4
        self.id_path = nn.Conv2d(self.dim_in, self.dim_out, 1) if self.dim_in != self.dim_out else nn.Identity()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.dim_in, self.dim_hid, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim_hid, self.dim_hid, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim_hid, self.dim_hid, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim_hid, self.dim_out, 1)
        )

    def forward(self, x):
        return self.id_path(x) + self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hid = dim_out // 4
        self.id_path = nn.Conv2d(self.dim_in, self.dim_out, 1) if self.dim_in != self.dim_out else nn.Identity()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.dim_in, self.dim_hid, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim_hid, self.dim_hid, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim_hid, self.dim_hid, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim_hid, self.dim_out, 3, 1, 1)
        )

    def forward(self, x):
        return self.id_path(x) + self.block(x)

class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, down_ratio, dim, K=512):
        super().__init__()
        if down_ratio == 4:  # A simpler version for MNIST with 4x downsampling ratio
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 4, 2, 1),
                ResBlock(dim),
                ResBlock(dim),
            )
            self.decoder = nn.Sequential(
                ResBlock(dim),
                ResBlock(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
                nn.Tanh()
            )
            self.codebook = VQEmbedding(K, dim)
        elif down_ratio == 8:
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dim, dim, 7, padding=3),
                EncoderBlock(dim, dim),
                nn.MaxPool2d(kernel_size=2),
                EncoderBlock(dim, dim),
                nn.MaxPool2d(kernel_size=2),
                EncoderBlock(dim, 2 * dim),
                nn.MaxPool2d(kernel_size=2),
                EncoderBlock(2 * dim, 4 * dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                DecoderBlock(4 * dim, 2 * dim),
                nn.Upsample(scale_factor=2, mode='nearest'),
                DecoderBlock(2 * dim, dim),
                nn.Upsample(scale_factor=2, mode='nearest'),
                DecoderBlock(dim, dim),
                nn.Upsample(scale_factor=2, mode='nearest'),
                DecoderBlock(dim, dim),
                nn.ReLU(),
                nn.Conv2d(dim, input_dim, 1),
                nn.Tanh()
            )
            self.codebook = VQEmbedding(K, 4 * dim)

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        # latents_emb = self.codebook.embedding(latents)
        return latents

    def decode(self, latents=None):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x
