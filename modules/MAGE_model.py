from collections import OrderedDict
import functools
import numpy as np
import torch
from torch import nn

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AxialAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout: float = 0.1, axial_dim: int = 1):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.dropout = nn.Dropout(dropout)
        self.axial_dim = axial_dim

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        '''
            :param x: [B, L, H, W, dim]
            :return: [B, L, H, W, dim]
        '''
        total_dim = x.shape
        axial_dim = self.axial_dim if self.axial_dim > 0 else (self.axial_dim + len(total_dim))
        last_two_dims = [axial_dim, len(total_dim) - 1]
        dims_rest = set(range(0, len(total_dim))) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        unpermutation = np.argsort(np.array(permutation)).tolist()

        x = x.permute(permutation).contiguous()
        permutation_shape = x.shape
        x = x.view(-1, x.shape[-2], x.shape[-1]).transpose(0, 1)  # axial_dim, -1, dim
        x = x + self.dropout(self.attention(self.ln_1(x)))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        x = x.transpose(0, 1).view(permutation_shape).permute(unpermutation).contiguous()
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None, need_weights=False):
        attn_mask = attn_mask.to(dtype=q.dtype, device=q.device) if attn_mask is not None else None
        x, att_weights = self.attn(self.ln_q(q), self.ln_kv(k), self.ln_kv(v), need_weights=need_weights, attn_mask=attn_mask)
        x = q + self.dropout(x)
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x, att_weights

class TransformerTextHead(nn.Module):
    def __init__(self, vocab_size: int, transformer_width: int, transformer_layers: int, transformer_heads: int,
                 output_dim: int, context_length: int, dropout: float = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.padding_idx = 0
        LayerClass = nn.TransformerEncoderLayer
        _layer = LayerClass(
            transformer_width,
            transformer_heads,
            dim_feedforward=transformer_width * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(_layer, transformer_layers)

        self.token_embedding = nn.Embedding(vocab_size, transformer_width, padding_idx=self.padding_idx)
        self.positions = nn.Embedding(context_length, transformer_width)
        self.layer_norm = nn.LayerNorm(transformer_width, eps=1e-8, elementwise_affine=True)
        self.dropout = nn.Dropout(p=dropout)
        self.ln_text_final = nn.LayerNorm(transformer_width)
        self.text_projection = nn.Linear(transformer_width, output_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, text: torch.Tensor, text_length: torch.Tensor):
        position_indices = self._create_position_indices(text)
        x = self.token_embedding(text)
        position_embeddings = self.positions(position_indices)
        x = self.layer_norm(x + position_embeddings)
        x = self.dropout(x)

        # Zero-out embeddings for positions which have padding tokens.
        # shape: (batch_size, max_caption_length, 1)
        token_mask = (text != self.padding_idx).unsqueeze(-1)
        # shape: (batch_size, max_caption_length, hidden_size)
        x = x * token_mask.type(x.dtype)
        ones = torch.ones_like(text)
        caption_mask = text_length.unsqueeze(1) < ones.cumsum(dim=1)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(
            x,
            mask=None,
            src_key_padding_mask=caption_mask,
        )
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_text_final(x)

        x = self.text_projection(x)

        return x

    @functools.lru_cache(maxsize=128)
    def _create_position_indices(self, tokens: torch.Tensor):

        # Create position indices of the same size as token indices.
        batch_size, max_caption_length = tokens.size()
        positions = torch.arange(
            max_caption_length, dtype=tokens.dtype, device=tokens.device
        )
        # shape: (batch_size, max_caption_length)
        positions = positions.unsqueeze(0).expand(batch_size, max_caption_length)
        return positions

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, stride_t=1, downsample=False, spectral=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=[stride_t, stride, stride], padding=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=16, num_channels=out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 3, 3), stride=[1, 1, 1], padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=16, num_channels=out_planes)
        self.downsample = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=[stride_t, stride, stride], padding=[1, 1, 1], bias=False),
            nn.GroupNorm(num_channels=out_planes, num_groups=16)) if downsample is True else None
        self.stride = stride

        if spectral:
            self.conv1 = torch.nn.utils.spectral_norm(self.conv1)
            self.conv2 = torch.nn.utils.spectral_norm(self.conv2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ADAIN2D(nn.Module):
    def __init__(self, num_features, z_dim):
        super().__init__()
        self.num_features = num_features
        self.norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        self.conv_mu = nn.Sequential(nn.Conv2d(z_dim, num_features, 3, 1, 1), nn.Conv2d(num_features, num_features, 3, 1, 1))
        self.conv_var = nn.Sequential(nn.Conv2d(z_dim, num_features, 3, 1, 1), nn.Conv2d(num_features, num_features, 3, 1, 1))

    def forward(self, x, y):
        out = self.norm(x)
        gamma = self.conv_mu(y)
        beta = self.conv_var(y)
        out = gamma * out + beta
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels,
                 model_channels,
                 out_channels,
                 frames_length,
                 layers,
                 dropout=0.,):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.frames_length = frames_length
        self.layers = layers
        self.in_linear = nn.Linear(in_channels, model_channels)

        scale = model_channels ** -0.5
        self.T_positional_embedding = nn.Parameter(scale * torch.randn(frames_length, 1, 1, model_channels))

        num_heads = model_channels // 32
        self.blocks = nn.ModuleList([])
        for i in range(layers):
            self.blocks.append(
                AxialAttentionBlock(model_channels, num_heads, axial_dim=i % 3 + 1, attn_mask=self.build_casual_attention_mask(self.frames_length) if (i % 3 + 1) == 1 else None, dropout=dropout),
            )
        self.ln_final = nn.Linear(model_channels, out_channels)
        self.initialize_parameters()

    def initialize_parameters(self):
        proj_std = (self.model_channels ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.model_channels ** -0.5
        fc_std = (2 * self.model_channels) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def build_casual_attention_mask(self, L):
        ############## lower triangle mask ##########
        mask = torch.empty(L, L)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, motion, imgs):
        x = torch.cat([motion.unsqueeze(1), imgs], 1)  # B, T, H, W, C
        x = self.in_linear(x)
        x = x + self.T_positional_embedding

        for layer in self.blocks:
            x = layer(x)

        out = self.ln_final(x[:, 1:])  # B, (T-1)HW, codebook_size

        return out

class MAGE(nn.Module):
    def __init__(self,
                 codebook_size: int,
                 frames_length: int,
                 resolution: int,
                 vision_width: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 dropout: float = 0.1,
                 randomness: bool = False,
                 ):
        super().__init__()

        self.context_length = context_length
        self.frames_length = frames_length
        self.vision_width = vision_width
        self.dropout = nn.Dropout(dropout)

        vision_heads = vision_width // 32

        self.text_encoder = TransformerTextHead(
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            output_dim=vision_width,
            context_length=context_length,
            dropout=dropout,
        )

        self.cross_attn = TransformerBlock(vision_width, vision_heads, dropout=dropout)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=vision_width, out_channels=vision_width, kernel_size=3, stride=1, padding=1,
                      bias=False)
        )
        self.randomness = randomness
        if self.randomness:
            self.conv3d = nn.Sequential(
                BasicBlock(vision_width, vision_width, stride=1, stride_t=2, downsample=True),
                BasicBlock(vision_width, vision_width, stride=1, stride_t=2, downsample=True),
                BasicBlock(vision_width, vision_width, stride=1, stride_t=2, downsample=True),
                BasicBlock(vision_width, vision_width, stride=1, stride_t=2, downsample=True),
            )
            self.conv_mu2 = nn.Conv2d(vision_width, 64, 3, 1, 1)
            self.conv_var2 = nn.Conv2d(vision_width, 64, 3, 1, 1)
            self.conv_d2 = nn.Conv2d(64, vision_width, kernel_size=3, stride=1, padding=1, bias=False)

            self.adain = ADAIN2D(vision_width, vision_width)

        scale = vision_width ** -0.5
        self.visual_token_embedding = nn.Embedding(codebook_size, vision_width)
        self.speed_embedding = nn.Parameter(scale * torch.randn(1, vision_width))
        self.H_positional_embedding = nn.Parameter(scale * torch.randn(1, resolution, 1, vision_width))
        self.W_positional_embedding = nn.Parameter(scale * torch.randn(1, 1, resolution, vision_width))

        self.decoder = Decoder(vision_width, vision_width, codebook_size, frames_length, layers=6, dropout=dropout)

        self.ln_final = nn.Linear(vision_width, codebook_size)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.visual_token_embedding.weight, std=0.02)
        if self.randomness:
            for m in self.conv3d:
                if isinstance(m, nn.Conv3d):
                    m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def reparameterize(self, emb):
        mu, logvar = self.conv_mu2(emb), self.conv_var2(emb)
        eps = torch.randn_like(logvar)
        std = logvar.mul(0.5).exp_()
        return eps.mul(std).add_(mu), mu, logvar  #

    def forward(self, lantent_img, text, speed, caption_len=None, test_flag=False):
        '''
        :param lantent_img: [batchsize, frames_length, height, width]
        :param text: [batchsize, context_length]
        :param speed: [batchsize]
        '''
        if caption_len is not None:
            text_feats_key = self.text_encoder(text, caption_len)
        else:
            text_feats_key = self.text_encoder(text)

        prior_img_emb = self.visual_token_embedding(lantent_img)
        B, L, H, W, C = prior_img_emb.size()
        prior_img = prior_img_emb[:, :(self.frames_length-1)].permute(0, 1, 4, 2, 3).contiguous().view(-1, C, H, W)
        prior_img = self.conv(prior_img).view(B, self.frames_length-1, C, H, W).permute(0, 1, 3, 4, 2).contiguous()  # B, L-1, H, W, C
        prior_img = prior_img + self.H_positional_embedding + self.W_positional_embedding
        first_img = prior_img[:, 0].view(B, -1, C).permute(1, 0, 2).contiguous()  # HW, B, C

        text_feats_key = text_feats_key.permute(1, 0, 2).contiguous()
        motion_anchor, _ = self.cross_attn(first_img, text_feats_key, text_feats_key, need_weights=True)
        motion_anchor = motion_anchor.permute(1, 0, 2).contiguous().view(B, H, W, C)

        if self.randomness:
            prior_vid = prior_img_emb[:, self.frames_length:]
            video_emb_prior = self.conv3d(prior_vid.permute(0, 4, 1, 2, 3).to(dtype=torch.float)).squeeze(2) # B, C, H, W
            video_emb, mu2, logvar2 = self.reparameterize(video_emb_prior)
            if test_flag:  # for test
                video_emb = torch.randn_like(video_emb)
            video_emb = self.conv_d2(video_emb)

            motion_anchor = self.adain(motion_anchor.permute(0, 3, 1, 2).contiguous(), video_emb)
            motion_anchor = motion_anchor.permute(0, 2, 3, 1).contiguous()

        speed_emb = speed.view(B, 1) @ self.speed_embedding
        motion_anchor = motion_anchor + speed_emb.unsqueeze(1).unsqueeze(1)
        predict = self.decoder(motion_anchor, prior_img)

        if self.randomness:
            mu2, logvar2 = mu2.reshape(mu2.size(0), -1), logvar2.reshape(logvar2.size(0), -1)
            loss2 = -0.5 * torch.mean(torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp(), axis=1))
            loss3 = torch.mean(torch.pow(torch.norm(speed_emb, dim=-1), 2))
            return predict, loss2, loss3
        else:
            return predict




