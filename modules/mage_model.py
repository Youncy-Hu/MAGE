from collections import OrderedDict

import functools
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_transformers
from utils.util import instantiate_from_config, default, zero_module

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AxialAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, axial_dim: int = 1):
        super().__init__()

        self.d_model = d_model
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.axial_dim = axial_dim

    def attention(self, x: torch.Tensor, attn_mask=None):
        attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask=None):
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
        x = x + self.dropout(self.attention(self.ln_1(x), attn_mask))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        x = x.transpose(0, 1).view(permutation_shape).permute(unpermutation).contiguous()
        return x

    def flops(self):
        resolution = 16
        length = 10
        token_num = resolution * resolution * length
        # token_num = resolution * resolution * (length-1) + 20 + 1
        if self.axial_dim == 1:
            flops = 3 * token_num * self.d_model * self.d_model   #QKV
            flops += token_num * length * self.d_model * 2 # QKtV
            flops += 2 * token_num * self.d_model * self.d_model * 4 #MLP
            flops += token_num * self.d_model * 2 #norm
        else:
            flops = 3 * token_num * self.d_model * self.d_model   #QKV
            flops += token_num * resolution * self.d_model * 2 # QKtV
            flops += 2 * token_num * self.d_model * self.d_model * 4 #MLP
            flops += token_num * self.d_model * 2 #norm
        return flops

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v, key_mask=None):
        key_mask = key_mask.to(device=q.device) if key_mask is not None else None
        return self.attn(q, k, v, need_weights=False, key_padding_mask=key_mask)[0]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_mask=None, need_weights=False):
        x = q + self.dropout(self.attention(q, k, v)) #NOTE: self.ln_q and self.ln_kv are not utilized in MAGE. Kindly comment out this line when employing MAGE+.
        # x = q + self.dropout(self.attention(self.ln_q(q), self.ln_kv(k), self.ln_kv(v), key_mask)) #NOTE: Please uncomment this line when employing MAGE+.
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x

    def flops(self, q_num=16*16, k_num=20):
        flops = k_num * self.d_model * self.d_model * 2 + q_num * self.d_model * self.d_model   #QKV
        flops += q_num * k_num * self.d_model * 2 # QKtV
        flops += 2 * q_num * self.d_model * self.d_model * 4 #MLP
        flops += q_num * self.d_model  #norm
        return flops

class MAEncoder(nn.Module):
    def __init__(self, layers: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = layers
        n_head = d_model // 32
        self.blocks = nn.ModuleList([])
        for _ in range(layers):
            self.blocks.append(TransformerBlock(d_model, n_head, dropout))

    def forward(self, x: torch.Tensor, kv: torch.Tensor, key_mask=None, need_weights=False):
        for i in range(self.layers):
            x = self.blocks[i](x, kv, kv, key_mask)
        return x

    def flops(self, q_num=16*16, k_num=20):
        flops = 0
        for i in range(self.layers):
            flops += self.blocks[i].flops(q_num, k_num)
        return flops

class BertTextualHead(nn.Module):
    def __init__(
        self,
        bert_path,
        out_dim: int,
    ):
        super().__init__()
        self.model_name = 'BertModel'
        self.pretrained_weights_name = bert_path
        self.model = getattr(
            pytorch_transformers,
            self.model_name).from_pretrained(
            self.pretrained_weights_name,
            output_hidden_states=True,
            output_attentions=True)
        self.model.training = True
        self.hidden_size = self.model.config.hidden_size
        self.text_out_feat_size = out_dim
        self.padding_idx = self.model.config.pad_token_id

        scale = self.hidden_size ** -0.5
        # self.text_projection = nn.Parameter(scale * torch.randn(self.hidden_size, out_dim))
        # self.text_projection = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.GELU(),
        #     nn.Linear(self.hidden_size, out_dim)
        # )
        self.text_projection_key = nn.Parameter(scale * torch.randn(self.hidden_size, out_dim))

    def forward(self, caption_tokens: torch.Tensor):
        """
        Bert Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        """
        output = self.model(caption_tokens)
        embed = output[0]

        # x = self.text_projection(embed[:, 0, :])
        x = embed @ self.text_projection_key

        return x

class TransformerTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, transformer_width: int, transformer_layers: int,
                 output_dim: int, context_length: int, padding_idx: int = 0, dropout: float = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.transformer_width = transformer_width
        self.context_length = context_length
        self.transformer_layers = transformer_layers
        self.output_dim = output_dim
        num_heads = transformer_width // 32
        LayerClass = nn.TransformerEncoderLayer
        _layer = LayerClass(
            transformer_width,
            num_heads,
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

    def forward(self, text: torch.Tensor):
        text_length = (text != self.padding_idx).float().sum(-1)
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
        # self.conv_mu = nn.Sequential(nn.Conv2d(z_dim, num_features, 7, 4, 3), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(num_features, num_features, 7, 4, 3))
        # self.conv_var = nn.Sequential(nn.Conv2d(z_dim, num_features, 7, 4, 3), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(num_features, num_features, 7, 4, 3))

    def forward(self, x, y):
        out = self.norm(x)
        gamma = self.conv_mu(y)
        beta = self.conv_var(y)
        out = gamma * out + beta
        return out


class FlatAxialDecoder(nn.Module):
    def __init__(self, in_channels,
                 model_channels,
                 out_channels,
                 frames_length,
                 layers,
                 context_channels=None,
                 use_cids=True,
                 dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.frames_length = frames_length
        self.layers = layers
        self.in_linear = nn.Linear(in_channels, model_channels)

        context_channels = default(context_channels, in_channels)
        self.context_linear = nn.Linear(context_channels, model_channels)

        scale = model_channels ** -0.5
        self.T_positional_embedding = nn.Parameter(scale * torch.randn(frames_length, 1, 1, model_channels))

        num_heads = model_channels // 32
        self.blocks = nn.ModuleList([])
        for i in range(layers):
            self.blocks.append(
                AxialAttentionBlock(model_channels, num_heads, axial_dim=i % 3 + 1, dropout=dropout,)
            )
        self.use_cids = use_cids
        if self.use_cids:
            self.out = nn.Linear(model_channels, out_channels)
        else:
            self.out = nn.Sequential(
                nn.GroupNorm(32, model_channels),
                nn.SiLU(),
                zero_module(nn.Conv3d(model_channels, out_channels, 1)),
            )
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

    def build_casual_attention_mask(self):
        ############## lower triangle mask ##########
        mask = torch.empty(self.frames_length, self.frames_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, motion, imgs):
        imgs = self.in_linear(imgs)
        motion = self.context_linear(motion)
        x = torch.cat([motion.unsqueeze(1), imgs], 1)
        x = x + self.T_positional_embedding

        temporal_mask = self.build_casual_attention_mask()
        for i, layer in enumerate(self.blocks):
            x = layer(x, attn_mask=temporal_mask if (i % 3)==0 else None)

        if self.use_cids:
            out = self.out(x[:, 1:])
        else:
            x = x[:, 1:].permute(0, 4, 1, 2, 3).contiguous()
            out = self.out(x).permute(0, 2, 3, 4, 1).contiguous()

        return out


from math import exp
class PIDControl():
    """docstring for ClassName"""

    def __init__(self):
        """define them out of loop"""
        # self.exp_KL = exp_KL
        self.I_k1 = 0.0
        self.W_k1 = 0.0
        self.e_k1 = 0.0

    def _Kp_fun(self, Err, scale=1):
        return 1.0 / (1.0 + float(scale) * exp(Err))

    def pid(self, exp_KL, KL_loss, Kp=0.01, Ki=-0.0001, Kd=0.0):
        """
        position PID algorithm
        Input: KL_loss
        return: weight for KL loss, beta
        """
        error_k = exp_KL - KL_loss
        ## comput U as the control factor
        Pk = Kp * self._Kp_fun(error_k)
        Ik = self.I_k1 + Ki * error_k
        # Dk = (error_k - self.e_k1) * Kd

        ## window up for integrator
        if self.W_k1 < 0 and self.W_k1 >= 1:
            Ik = self.I_k1

        Wk = Pk + Ik
        self.W_k1 = Wk
        self.I_k1 = Ik
        self.e_k1 = error_k

        ## min and max value
        if Wk >= 1:
            Wk = 1.0
        if Wk < 0:
            Wk = 0.0

        return Wk, error_k


from ldm.models.autoencoder import DiagonalGaussianDistribution
from modules.vqvae_model import VectorQuantizedVAE

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class MAGE(nn.Module):
    def __init__(self,
                 first_stage_config,
                 text_encoder_config,
                 ma_config,
                 generate_decoder_config,
                 codebook_size: int,
                 frames_length: int,
                 image_resolution: int,
                 vision_width: int,
                 dropout: float = 0.1,
                 use_cids=False,
                 randomness=False,
                 alpha=0.,
                 beta=1.,
                 v_kl=0.,
                 auto_beta=False,
                 ):
        super().__init__()

        self.instantiate_first_stage(first_stage_config)
        self.frames_length = frames_length
        self.image_resolution = image_resolution
        self.vision_width = vision_width
        self.dropout = dropout
        self.use_cids = use_cids
        self.auto_beta = auto_beta

        self.text_encoder = instantiate_from_config(text_encoder_config)
        self.ma_encoder = instantiate_from_config(ma_config, {'dropout': dropout})
        self.generate_model = instantiate_from_config(generate_decoder_config,
                                                      {'use_cids': use_cids, 'dropout': dropout, 'context_channels': ma_config['params']['d_model']})

        self.codebook_size = codebook_size
        if use_cids:
            self.visual_token_embedding = nn.Embedding(codebook_size, vision_width)
        else:
            self.visual_token_embedding = nn.Linear(self.first_stage_model.embed_dim, vision_width)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=vision_width, out_channels=vision_width, kernel_size=3, stride=1, padding=1,
                      bias=False)
        )
        scale = vision_width ** -0.5
        self.speed_embedding = nn.Parameter(scale * torch.randn(1, vision_width))
        self.H_positional_embedding = nn.Parameter(scale * torch.randn(1, image_resolution, 1, vision_width))
        self.W_positional_embedding = nn.Parameter(scale * torch.randn(1, 1, image_resolution, vision_width))

        self.randomness = randomness
        if self.randomness:
            self.conv3d = nn.Sequential(
                BasicBlock(vision_width, vision_width, stride=1, stride_t=2, downsample=True),
                BasicBlock(vision_width, vision_width, stride=1, stride_t=2, downsample=True),
                BasicBlock(vision_width, vision_width, stride=1, stride_t=2, downsample=True),
                BasicBlock(vision_width, ma_config['params']['d_model'], stride=1, stride_t=2, downsample=True),
            )
            self.conv_mu2 = nn.Conv2d(vision_width, 64, 3, 1, 1)
            self.conv_var2 = nn.Conv2d(vision_width, 64, 3, 1, 1)
            self.conv_d2 = nn.Conv2d(64, vision_width, kernel_size=3, stride=1, padding=1, bias=False)
            self.adain = ADAIN2D(vision_width, vision_width)

            if self.auto_beta:
                self.PID = PIDControl()
                self.KL_loss = v_kl
            else:
                self.alpha = alpha
                self.beta = beta

        self.initialize_parameters()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def initialize_parameters(self):
        nn.init.normal_(self.visual_token_embedding.weight, std=0.02)
        if self.randomness:
            for m in self.conv3d:
                if isinstance(m, nn.Conv3d):
                    m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

    @torch.no_grad()
    def first_stage_encode(self, x):
        '''
        :param x: B, T, C, H, W
        :return: B, T, h, w if use_cids else B, T, c, h, w
        '''
        out = x.view(-1, *x.size()[-3:])
        encoder_posterior = self.first_stage_model.encode(out)
        out = self.get_first_stage_encoding(encoder_posterior)
        out = out.view(*x.size()[:-3], *out.size()[1:]).contiguous().detach()
        return out

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return z

    @torch.no_grad()
    def first_stage_decode(self, x):
        '''
        :param x: B, T, h, w if use_cids else B, T, c, h, w
        :return: B, T, C, H, W
        '''
        if not self.use_cids:
            out = x.view(-1, *x.size()[-3:])
        else:
            out = x.view(-1, *x.size()[-2:])

        if isinstance(self.first_stage_model, VectorQuantizedVAE):
            out = self.first_stage_model.decode(out)
        else:
            out = self.first_stage_model.decode(out)
        out = out.view(*x.size()[:2], *out.size()[1:]).contiguous().detach()
        return out

    def reparameterize(self, emb):
        mu, logvar = self.conv_mu2(emb), self.conv_var2(emb)
        eps = torch.randn_like(logvar)
        std = logvar.mul(0.5).exp_()
        return eps.mul(std).add_(mu), mu, logvar

    def forward(self, batch, test_flag=False):

        if 'images' in batch.keys():
            x = batch['images']
        x = self.first_stage_encode(x)
        if self.use_cids:
            x_emb = self.visual_token_embedding(x.to(torch.long)).permute(0, 1, 4, 2, 3).contiguous()
        else:
            x_emb = self.visual_token_embedding(x.permute(0, 1, 3, 4, 2).contiguous()).permute(0, 1, 4, 2, 3).contiguous()
        B, L, C, H, W = x_emb.size()

        prior_img = x_emb[:, :(self.frames_length-1)].reshape(-1, C, H, W)
        prior_img = self.conv(prior_img).view(B, -1, C, H, W).permute(0, 1, 3, 4, 2).contiguous()
        prior_img = prior_img + self.H_positional_embedding + self.W_positional_embedding
        first_img = prior_img[:, 0].reshape(B, -1, C).permute(1, 0, 2).contiguous()

        text_emb = self.text_encoder(batch['text']).permute(1, 0, 2).contiguous()
        # text_length = (batch['text'] != self.text_encoder.padding_idx).float().sum(-1)
        # ones = torch.ones_like(batch['text'])
        # caption_mask = text_length.unsqueeze(1) < ones.cumsum(dim=1)

        motion_anchor = self.ma_encoder(first_img, text_emb, need_weights=False)  #, key_mask=caption_mask
        motion_anchor = motion_anchor.permute(1, 0, 2).contiguous().view(B, H, W, C)

        prefix = 'train' if self.training else 'val'
        if self.randomness:
            prior_vid = x_emb.permute(0, 2, 1, 3, 4).contiguous()
            video_emb_prior = self.conv3d(prior_vid).squeeze(2) # B, C, H, W
            video_emb, mu2, logvar2 = self.reparameterize(video_emb_prior)
            if test_flag:  # for test
                video_emb = torch.randn_like(video_emb)
            video_emb = self.conv_d2(video_emb)

            motion_anchor = self.adain(motion_anchor.permute(0, 3, 1, 2).contiguous(), video_emb)
            motion_anchor = motion_anchor.permute(0, 2, 3, 1).contiguous()

        if 'speed' in batch.keys():
            speed_emb = batch['speed'].view(B, 1) @ self.speed_embedding
            motion_anchor = motion_anchor + speed_emb.unsqueeze(1).unsqueeze(1)

        model_predict = self.generate_model(motion_anchor, prior_img)
        loss_dict = {}
        if self.use_cids:
            recon_loss = F.cross_entropy(model_predict.view(-1, self.codebook_size), x[:, 1:self.frames_length].contiguous().view(-1).to(torch.long))
        else:
            recon_loss = F.mse_loss(model_predict.permute(0, 1, 4, 2, 3).contiguous(), x[:, 1:])
        loss_dict.update({f'{prefix}/prediction': recon_loss.clone().detach().item()})

        if self.randomness:
            mu2, logvar2 = mu2.reshape(mu2.size(0), -1), logvar2.reshape(logvar2.size(0), -1)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp(), axis=1))
            loss_dict.update({f'{prefix}/kl_loss': kl_loss.clone().detach().item()})

            if self.auto_beta:
                self.beta, _ = self.PID.pid(self.KL_loss, kl_loss.item())
                loss_dict.update({f'{prefix}/beta': self.beta})
                final_loss = recon_loss + self.beta * kl_loss
            else:
                l2_loss = torch.mean(torch.pow(torch.norm(speed_emb, dim=-1), 2)) # This loss is to align with MAGE. We discarded it in MAGE+.
                final_loss = recon_loss + self.beta * kl_loss + self.alpha * l2_loss
        else:
            final_loss = recon_loss
        loss_dict.update({f'{prefix}/final_loss': final_loss.clone().detach().item()})

        return final_loss, loss_dict

    def autoregressive_generate(self, batch):
        x = self.first_stage_encode(batch['images'][:, 0:1])
        if self.use_cids:
            x_emb = self.visual_token_embedding(x.to(torch.long)).permute(0, 1, 4, 2, 3).contiguous()
        else:
            x_emb = self.visual_token_embedding(x.permute(0, 1, 3, 4, 2).contiguous()).permute(0, 1, 4, 2, 3).contiguous()
        B, _, C, H, W = x_emb.size()
        first_img = self.conv(x_emb.reshape(-1, C, H, W)).view(B, -1, C, H, W).permute(0, 1, 3, 4, 2).contiguous()
        first_img = first_img + self.H_positional_embedding + self.W_positional_embedding
        first_img = first_img[:, 0].reshape(B, -1, C).permute(1, 0, 2).contiguous()  # HW, B, C

        text_emb = self.text_encoder(batch['text']).permute(1, 0, 2).contiguous()
        # text_length = (batch['text'] != self.text_encoder.padding_idx).float().sum(-1)
        # ones = torch.ones_like(batch['text'])
        # caption_mask = text_length.unsqueeze(1) < ones.cumsum(dim=1)

        motion_anchor = self.ma_encoder(first_img, text_emb, need_weights=False) #, key_mask=caption_mask
        motion_anchor = motion_anchor.permute(1, 0, 2).contiguous().view(B, H, W, C)

        if self.randomness:
            video_emb = torch.randn([B, 64, H, W]).to(motion_anchor.device)
            video_emb = self.conv_d2(video_emb)
            motion_anchor = self.adain(motion_anchor.permute(0, 3, 1, 2).contiguous(), video_emb)
            motion_anchor = motion_anchor.permute(0, 2, 3, 1).contiguous()

        if 'speed' in batch.keys():
            speed_emb = batch['speed'].view(B, 1) @ self.speed_embedding
            motion_anchor = motion_anchor + speed_emb.unsqueeze(1).unsqueeze(1)

        input = x_emb.repeat(1, self.frames_length-1, 1, 1, 1)
        B, L, C, H, W = input.size()

        for i in range(self.frames_length - 1):
            imgs_emb = input.view(-1, C, H, W)
            imgs_emb = self.conv(imgs_emb).view(B, -1, C, H, W).permute(0, 1, 3, 4, 2).contiguous()
            imgs_emb = imgs_emb + self.H_positional_embedding + self.W_positional_embedding
            prediction = self.generate_model(motion_anchor, imgs_emb)

            if i != self.frames_length - 2:
                if self.use_cids:
                    prediction_id = torch.max(prediction, -1)[1]
                    input[:, i + 1] = self.visual_token_embedding(prediction_id[:, i, :, :]).permute(0, 3, 1, 2)
                else:
                    input[:, i + 1] = self.visual_token_embedding(prediction).permute(0, 1, 4, 2, 3)[:, i]

        if self.use_cids:
            gen_video = torch.max(prediction, -1)[1]
        else:
            gen_video = prediction.permute(0, 1, 4, 2, 3).contiguous()
        gen_video = self.first_stage_decode(gen_video)
        gen_video = torch.cat([batch['images'][:, 0:1], gen_video], 1)

        return gen_video
