# ComfyUI-native DiffuEraser core.
# Self-contained reimplementation of BrushNet + UNetMotionModel (AnimateDiff-style
# motion modules) + TCD scheduler + the DiffuEraser clip-window sampling loop.
#
# Design notes (mirrors the TE/TwinFlow core-wrapper pattern):
#   - zero diffusers / transformers imports; keys stay 1:1 with the diffusers
#     checkpoints so weights load via load_state_dict(strict=True)
#   - weights live on comfy ops (manual_cast) so comfy.model_management /
#     ModelPatcher handle device/dtype/offload (replaces group offloading hacks)
#   - attention math delegates to comfy.ldm.modules.attention.optimized_attention
#   - VAE comes from comfy.sd.VAE (raw latents in/out, scaling handled here)
#   - numerics replicate the vendored diffusers-0.29-era code that DiffuEraser was
#     trained against, including known quirks (see UNetMidBlockCrossAttnMotion)

import json
import math
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

import comfy.ops
import comfy.model_patcher
import comfy.model_management as mm
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.utils import load_torch_file
import comfy.sd

VAE_SCALE_FACTOR = 0.18215  # SD1.5


# ---------------------------------------------------------------------------
# basic embeddings
# ---------------------------------------------------------------------------

def get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0, scale=1.0):
    half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels, flip_sin_to_cos, downscale_freq_shift, scale=1.0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        return get_timestep_embedding(timesteps, self.num_channels, self.flip_sin_to_cos,
                                      self.downscale_freq_shift, self.scale)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim, act_fn="silu", out_dim=None, operations=comfy.ops.manual_cast):
        super().__init__()
        self.linear_1 = operations.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        out_dim = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = operations.Linear(time_embed_dim, out_dim)

    def forward(self, sample, condition=None):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class SinusoidalPositionalEmbedding(nn.Module):
    # fixed (non-learned) table, so no state-dict keys
    def __init__(self, embed_dim, max_seq_length=32):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # persistent: pe is stored in the checkpoints

    def forward(self, x):
        seq_length = x.shape[1]
        # comfy weight loading moves parameters but not this buffer; move lazily
        pe = self.pe.to(device=x.device, dtype=x.dtype)
        return x + pe[:, :seq_length]


# ---------------------------------------------------------------------------
# attention / transformer primitives (keys match diffusers 1:1)
# ---------------------------------------------------------------------------

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, operations=comfy.ops.manual_cast):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out * 2, bias=bias)
        self.gelu = nn.GELU()

    def gelu_2(self, x):
        return x * self.gelu(x)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.0, operations=comfy.ops.manual_cast):
        super().__init__()
        inner_dim = 4 * dim
        self.net = nn.ModuleList([
            GEGLU(dim, inner_dim, bias=True, operations=operations),
            nn.Dropout(dropout),
            operations.Linear(inner_dim, dim, bias=True),
        ])

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class Attention(nn.Module):
    # covers the SD1.5 spatial attention and the double-self-attention used by
    # AnimateDiff motion modules; optional cross_attention_dim
    def __init__(self, query_dim, heads, dim_head, bias=False, cross_attention_dim=None, operations=comfy.ops.manual_cast):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = operations.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = operations.Linear(cross_attention_dim or query_dim, inner_dim, bias=bias)
        self.to_v = operations.Linear(cross_attention_dim or query_dim, inner_dim, bias=bias)
        # diffusers: out_bias defaults to True independently of attention bias
        self.to_out = nn.ModuleList([operations.Linear(inner_dim, query_dim, bias=True), nn.Dropout(0.0)])

    def forward(self, hidden_states, encoder_hidden_states=None):
        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        v = self.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        fn = optimized_attention_for_device(q.device)
        out = fn(q, k, v, self.heads)
        return self.to_out[0](out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.0, cross_attention_dim=None, activation_fn="geglu",
                 attention_bias=False, double_self_attention=False, norm_elementwise_affine=True,
                 positional_embeddings=None, num_positional_embeddings=None, operations=comfy.ops.manual_cast):
        super().__init__()
        self.attn1 = Attention(dim, heads, dim_head, bias=attention_bias,
                               cross_attention_dim=cross_attention_dim if double_self_attention else None,
                               operations=operations)
        self.norm1 = operations.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=1e-5)
        if double_self_attention:
            # diffusers: second Attention(query_dim) with self-attention
            self.attn2 = Attention(dim, heads, dim_head, bias=attention_bias,
                                   cross_attention_dim=None, operations=operations)
        elif cross_attention_dim is not None:
            self.attn2 = Attention(dim, heads, dim_head, bias=attention_bias,
                                   cross_attention_dim=cross_attention_dim, operations=operations)
        else:
            self.attn2 = None
        self.norm2 = operations.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=1e-5)
        self.norm3 = operations.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=1e-5)
        self.ff = FeedForward(dim, dropout=dropout, operations=operations)

        self.pos_embed = None
        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)

    def forward(self, hidden_states, encoder_hidden_states=None):
        norm_hidden_states = self.norm1(hidden_states)
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)
        hidden_states = self.attn1(norm_hidden_states) + hidden_states

        # 2. Cross-Attention (may be None for pure self-attention blocks)
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)
            hidden_states = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states) + hidden_states

        norm_hidden_states = self.norm3(hidden_states)
        hidden_states = self.ff(norm_hidden_states) + hidden_states
        return hidden_states


class Transformer2DModel(nn.Module):
    # SD1.5 spatial transformer: GroupNorm + 1x1 conv proj, blocks, 1x1 conv out
    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1,
                 cross_attention_dim=768, norm_num_groups=32, use_linear_projection=False,
                 only_cross_attention=False, upcast_attention=False, operations=comfy.ops.manual_cast):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.use_linear_projection = use_linear_projection
        self.only_cross_attention = only_cross_attention

        self.norm = operations.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = operations.Linear(in_channels, inner_dim)
        else:
            self.proj_in = operations.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, num_attention_heads, attention_head_dim,
                                  dropout=0.0, cross_attention_dim=cross_attention_dim,
                                  activation_fn="geglu", attention_bias=False,
                                  double_self_attention=False, norm_elementwise_affine=True,
                                  positional_embeddings=None, num_positional_embeddings=None,
                                  operations=operations)
            for _ in range(num_layers)
        ])

        if use_linear_projection:
            self.proj_out = operations.Linear(in_channels, inner_dim)
        else:
            self.proj_out = operations.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None):
        residual = hidden_states
        b, c, h, w = hidden_states.shape
        inner = self.norm(hidden_states)
        if self.use_linear_projection:
            inner = self.proj_in(inner.permute(0, 2, 3, 1).reshape(b, h * w, c))
        else:
            inner = self.proj_in(inner)
            inner = inner.permute(0, 2, 3, 1).reshape(b, h * w, c)

        for block in self.transformer_blocks:
            if self.only_cross_attention and encoder_hidden_states is not None:
                block_out = block(inner, encoder_hidden_states=encoder_hidden_states)
            else:
                block_out = block(inner, encoder_hidden_states=encoder_hidden_states)
            inner = block_out

        if self.use_linear_projection:
            inner = self.proj_out(inner)
            inner = inner.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        else:
            inner = inner.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            inner = self.proj_out(inner)

        return inner + residual


class TransformerTemporalModel(nn.Module):
    # AnimateDiff-style motion module (as used by diffusers UNetMotionModel)
    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1,
                 cross_attention_dim=None, norm_num_groups=32, attention_bias=False,
                 activation_fn="geglu", positional_embeddings=None, num_positional_embeddings=None,
                 operations=comfy.ops.manual_cast):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels

        self.norm = operations.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = operations.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, num_attention_heads, attention_head_dim,
                                  dropout=0.0, cross_attention_dim=cross_attention_dim,
                                  activation_fn=activation_fn, attention_bias=attention_bias,
                                  double_self_attention=True, norm_elementwise_affine=True,
                                  positional_embeddings=positional_embeddings,
                                  num_positional_embeddings=num_positional_embeddings,
                                  operations=operations)
            for _ in range(num_layers)
        ])
        self.proj_out = operations.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, num_frames=1):
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames
        residual = hidden_states

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = (hidden_states[None, None, :]
                         .reshape(batch_size, height, width, num_frames, channel)
                         .permute(0, 3, 4, 1, 2)
                         .contiguous())
        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)
        return hidden_states + residual


# ---------------------------------------------------------------------------
# conv blocks
# ---------------------------------------------------------------------------

class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, eps=1e-6, groups=32, dropout=0.0,
                 non_linearity="silu", output_scale_factor=1.0, operations=comfy.ops.manual_cast):
        super().__init__()
        self.output_scale_factor = output_scale_factor
        self.norm1 = operations.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = operations.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels is not None:
            self.time_emb_proj = operations.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None
        self.norm2 = operations.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = operations.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        if in_channels != out_channels:
            self.conv_shortcut = operations.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None

    def forward(self, input_tensor, temb, scale=1.0):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        if self.time_emb_proj is not None:
            temb_out = self.nonlinearity(temb)
            temb_out = self.time_emb_proj(temb_out)[:, :, None, None]
            hidden_states = hidden_states + temb_out
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        return (input_tensor + hidden_states) / self.output_scale_factor


class Downsample2D(nn.Module):
    def __init__(self, channels, use_conv=True, out_channels=None, padding=1, name="op", operations=comfy.ops.manual_cast):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        if use_conv:
            self.conv = operations.Conv2d(self.channels, self.out_channels, kernel_size=3, stride=stride, padding=padding)
        else:
            self.conv = nn.AvgPool2d(kernel_size=stride, stride=stride)
        self.name = name  # diffusers checkpoints store downsampler weights as .conv.*

    def forward(self, hidden_states, scale=1.0):
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    def __init__(self, channels, use_conv=True, out_channels=None, name="conv", operations=comfy.ops.manual_cast):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = operations.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, hidden_states, upsample_size=None, scale=1.0):
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        if upsample_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=upsample_size, mode="nearest")
        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# motion (3D) blocks
# ---------------------------------------------------------------------------

class DownBlockMotion(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, num_layers=2, resnet_eps=1e-6,
                 resnet_act_fn="silu", resnet_groups=32, add_downsample=True, downsample_padding=1,
                 temporal_num_attention_heads=8, temporal_max_seq_length=32, operations=comfy.ops.manual_cast):
        super().__init__()
        resnets, motion_modules = [], []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_c, out_channels, temb_channels, eps=resnet_eps,
                                         groups=resnet_groups, non_linearity=resnet_act_fn, operations=operations))
            motion_modules.append(TransformerTemporalModel(
                num_attention_heads=temporal_num_attention_heads,
                attention_head_dim=out_channels // temporal_num_attention_heads,
                in_channels=out_channels, norm_num_groups=resnet_groups, cross_attention_dim=None,
                attention_bias=False, activation_fn="geglu", positional_embeddings="sinusoidal",
                num_positional_embeddings=temporal_max_seq_length, operations=operations))
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels,
                                                            padding=downsample_padding, name="op", operations=operations)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, down_block_add_samples=None, temb=None, num_frames=1):
        output_states = ()
        for resnet, motion_module in zip(self.resnets, self.motion_modules):
            hidden_states = resnet(hidden_states, temb)
            if down_block_add_samples is not None:
                hidden_states = hidden_states + down_block_add_samples.pop(0)
            hidden_states = motion_module(hidden_states, num_frames=num_frames)
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            if down_block_add_samples is not None:
                hidden_states = hidden_states + down_block_add_samples.pop(0)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class CrossAttnDownBlockMotion(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, num_layers=2, resnet_eps=1e-6,
                 resnet_act_fn="silu", resnet_groups=32, cross_attention_dim=768, num_attention_heads=8,
                 downsample_padding=1, add_downsample=True, use_linear_projection=False,
                 only_cross_attention=False, temporal_num_attention_heads=8, temporal_max_seq_length=32,
                 operations=comfy.ops.manual_cast):
        super().__init__()
        self.has_cross_attention = True
        resnets, attentions, motion_modules = [], [], []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_c, out_channels, temb_channels, eps=resnet_eps,
                                         groups=resnet_groups, non_linearity=resnet_act_fn, operations=operations))
            attentions.append(Transformer2DModel(
                num_attention_heads, out_channels // num_attention_heads, in_channels=out_channels,
                num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups,
                use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention,
                operations=operations))
            motion_modules.append(TransformerTemporalModel(
                num_attention_heads=temporal_num_attention_heads,
                attention_head_dim=out_channels // temporal_num_attention_heads,
                in_channels=out_channels, norm_num_groups=resnet_groups, cross_attention_dim=None,
                attention_bias=False, activation_fn="geglu", positional_embeddings="sinusoidal",
                num_positional_embeddings=temporal_max_seq_length, operations=operations))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels,
                                                            padding=downsample_padding, name="op", operations=operations)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, num_frames=1,
                additional_residuals=None, down_block_add_samples=None):
        output_states = ()
        blocks = list(zip(self.resnets, self.attentions, self.motion_modules))
        for i, (resnet, attn, motion_module) in enumerate(blocks):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals
            if down_block_add_samples is not None:
                hidden_states = hidden_states + down_block_add_samples.pop(0)
            hidden_states = motion_module(hidden_states, num_frames=num_frames)
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            if down_block_add_samples is not None:
                hidden_states = hidden_states + down_block_add_samples.pop(0)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class UNetMidBlockCrossAttnMotion(nn.Module):
    def __init__(self, in_channels, temb_channels, num_layers=1, resnet_eps=1e-6, resnet_act_fn="silu",
                 resnet_groups=32, num_attention_heads=8, cross_attention_dim=768,
                 temporal_num_attention_heads=8, temporal_max_seq_length=32, operations=comfy.ops.manual_cast):
        super().__init__()
        self.has_cross_attention = True
        resnets = [ResnetBlock2D(in_channels, in_channels, temb_channels, eps=resnet_eps,
                                 groups=resnet_groups, non_linearity=resnet_act_fn, operations=operations)]
        attentions, motion_modules = [], []
        for _ in range(num_layers):
            attentions.append(Transformer2DModel(
                num_attention_heads, in_channels // num_attention_heads, in_channels=in_channels,
                num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups,
                use_linear_projection=False, operations=operations))
            resnets.append(ResnetBlock2D(in_channels, in_channels, temb_channels, eps=resnet_eps,
                                         groups=resnet_groups, non_linearity=resnet_act_fn, operations=operations))
            motion_modules.append(TransformerTemporalModel(
                num_attention_heads=temporal_num_attention_heads,
                attention_head_dim=in_channels // temporal_num_attention_heads,
                in_channels=in_channels, norm_num_groups=resnet_groups, cross_attention_dim=None,
                attention_bias=False, positional_embeddings="sinusoidal",
                num_positional_embeddings=temporal_max_seq_length, activation_fn="geglu",
                operations=operations))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, num_frames=1, mid_block_add_sample=None):
        # NOTE: preserves the vendored DiffuEraser quirk of applying the second
        # resnet twice (before and after the motion module) — weights were
        # trained against this exact flow.
        lora_scale = 1.0
        hidden_states = self.resnets[0](hidden_states, temb, scale=lora_scale)
        for attn, resnet, motion_module in zip(self.attentions, self.resnets[1:], self.motion_modules):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb, scale=lora_scale)
            if mid_block_add_sample is not None:
                hidden_states = hidden_states + mid_block_add_sample
            hidden_states = motion_module(hidden_states, num_frames=num_frames)
            hidden_states = resnet(hidden_states, temb, scale=lora_scale)
        return hidden_states


class CrossAttnUpBlockMotion(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, temb_channels, num_layers=3,
                 resnet_eps=1e-6, resnet_act_fn="silu", resnet_groups=32, cross_attention_dim=768,
                 num_attention_heads=8, add_upsample=True, use_linear_projection=False,
                 only_cross_attention=False, temporal_num_attention_heads=8, temporal_max_seq_length=32,
                 resolution_idx=None, operations=comfy.ops.manual_cast):
        super().__init__()
        self.has_cross_attention = True
        resnets, attentions, motion_modules = [], [], []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(resnet_in_channels + res_skip_channels, out_channels, temb_channels,
                                         eps=resnet_eps, groups=resnet_groups, non_linearity=resnet_act_fn,
                                         operations=operations))
            attentions.append(Transformer2DModel(
                num_attention_heads, out_channels // num_attention_heads, in_channels=out_channels,
                num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups,
                use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention,
                operations=operations))
            motion_modules.append(TransformerTemporalModel(
                num_attention_heads=temporal_num_attention_heads,
                attention_head_dim=out_channels // temporal_num_attention_heads,
                in_channels=out_channels, norm_num_groups=resnet_groups, cross_attention_dim=None,
                attention_bias=False, activation_fn="geglu", positional_embeddings="sinusoidal",
                num_positional_embeddings=temporal_max_seq_length, operations=operations))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels,
                                                        operations=operations)])
        else:
            self.upsamplers = None
        self.resolution_idx = resolution_idx

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, encoder_hidden_states=None,
                upsample_size=None, num_frames=1, up_block_add_samples=None):
        for resnet, attn, motion_module in zip(self.resnets, self.attentions, self.motion_modules):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            if up_block_add_samples is not None:
                hidden_states = hidden_states + up_block_add_samples.pop(0)
            hidden_states = motion_module(hidden_states, num_frames=num_frames)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
            if up_block_add_samples is not None:
                hidden_states = hidden_states + up_block_add_samples.pop(0)
        return hidden_states


class UpBlockMotion(nn.Module):
    def __init__(self, in_channels, prev_output_channel, out_channels, temb_channels, num_layers=3,
                 resnet_eps=1e-6, resnet_act_fn="silu", resnet_groups=32, add_upsample=True,
                 temporal_num_attention_heads=8, temporal_max_seq_length=32, resolution_idx=None,
                 operations=comfy.ops.manual_cast):
        super().__init__()
        resnets, motion_modules = [], []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(resnet_in_channels + res_skip_channels, out_channels, temb_channels,
                                         eps=resnet_eps, groups=resnet_groups, non_linearity=resnet_act_fn,
                                         operations=operations))
            motion_modules.append(TransformerTemporalModel(
                num_attention_heads=temporal_num_attention_heads,
                attention_head_dim=out_channels // temporal_num_attention_heads,
                in_channels=out_channels, norm_num_groups=resnet_groups, cross_attention_dim=None,
                attention_bias=False, activation_fn="geglu", positional_embeddings="sinusoidal",
                num_positional_embeddings=temporal_max_seq_length, operations=operations))
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels,
                                                        operations=operations)])
        else:
            self.upsamplers = None
        self.resolution_idx = resolution_idx

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None,
                num_frames=1, up_block_add_samples=None):
        for resnet, motion_module in zip(self.resnets, self.motion_modules):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            if up_block_add_samples is not None:
                hidden_states = hidden_states + up_block_add_samples.pop(0)
            hidden_states = motion_module(hidden_states, num_frames=num_frames)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
            if up_block_add_samples is not None:
                hidden_states = hidden_states + up_block_add_samples.pop(0)
        return hidden_states


# ---------------------------------------------------------------------------
# 2D blocks (BrushNet side)
# ---------------------------------------------------------------------------

class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, num_layers=2, resnet_eps=1e-6,
                 resnet_act_fn="silu", resnet_groups=32, add_downsample=True, downsample_padding=1,
                 operations=comfy.ops.manual_cast):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_c, out_channels, temb_channels, eps=resnet_eps,
                                         groups=resnet_groups, non_linearity=resnet_act_fn, operations=operations))
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels,
                                                            padding=downsample_padding, name="op", operations=operations)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, down_block_add_samples=None):
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            if down_block_add_samples is not None:
                hidden_states = hidden_states + down_block_add_samples.pop(0)
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            if down_block_add_samples is not None:
                hidden_states = hidden_states + down_block_add_samples.pop(0)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, num_layers=2, resnet_eps=1e-6,
                 resnet_act_fn="silu", resnet_groups=32, cross_attention_dim=768, num_attention_heads=8,
                 downsample_padding=1, add_downsample=True, use_linear_projection=False,
                 only_cross_attention=False, operations=comfy.ops.manual_cast):
        super().__init__()
        self.has_cross_attention = True
        resnets, attentions = [], []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_c, out_channels, temb_channels, eps=resnet_eps,
                                         groups=resnet_groups, non_linearity=resnet_act_fn, operations=operations))
            attentions.append(Transformer2DModel(
                num_attention_heads, out_channels // num_attention_heads, in_channels=out_channels,
                num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups,
                use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention,
                operations=operations))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels,
                                                            padding=downsample_padding, name="op", operations=operations)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, down_block_add_samples=None):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            if down_block_add_samples is not None:
                hidden_states = hidden_states + down_block_add_samples.pop(0)
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            if down_block_add_samples is not None:
                hidden_states = hidden_states + down_block_add_samples.pop(0)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, in_channels, temb_channels, num_layers=1, resnet_eps=1e-6, resnet_act_fn="silu",
                 resnet_groups=32, num_attention_heads=8, cross_attention_dim=768,
                 operations=comfy.ops.manual_cast):
        super().__init__()
        self.has_cross_attention = True
        resnets = [ResnetBlock2D(in_channels, in_channels, temb_channels, eps=resnet_eps,
                                 groups=resnet_groups, non_linearity=resnet_act_fn, operations=operations)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(Transformer2DModel(
                num_attention_heads, in_channels // num_attention_heads, in_channels=in_channels,
                num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups,
                use_linear_projection=False, operations=operations))
            resnets.append(ResnetBlock2D(in_channels, in_channels, temb_channels, eps=resnet_eps,
                                         groups=resnet_groups, non_linearity=resnet_act_fn, operations=operations))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(self, in_channels, prev_output_channel, out_channels, temb_channels, num_layers=3,
                 resnet_eps=1e-6, resnet_act_fn="silu", resnet_groups=32, add_upsample=True,
                 resolution_idx=None, operations=comfy.ops.manual_cast):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(resnet_in_channels + res_skip_channels, out_channels, temb_channels,
                                         eps=resnet_eps, groups=resnet_groups, non_linearity=resnet_act_fn,
                                         operations=operations))
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels,
                                                        operations=operations)])
        else:
            self.upsamplers = None
        self.resolution_idx = resolution_idx

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None,
                return_res_samples=False, up_block_add_samples=None):
        output_states = () if return_res_samples else None
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            if return_res_samples:
                output_states = output_states + (hidden_states,)
            if up_block_add_samples is not None:
                hidden_states = hidden_states + up_block_add_samples.pop(0)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
            if return_res_samples:
                output_states = output_states + (hidden_states,)
            if up_block_add_samples is not None:
                hidden_states = hidden_states + up_block_add_samples.pop(0)
        if return_res_samples:
            return hidden_states, output_states
        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, temb_channels, num_layers=3,
                 resnet_eps=1e-6, resnet_act_fn="silu", resnet_groups=32, cross_attention_dim=768,
                 num_attention_heads=8, add_upsample=True, use_linear_projection=False,
                 only_cross_attention=False, resolution_idx=None, operations=comfy.ops.manual_cast):
        super().__init__()
        self.has_cross_attention = True
        resnets, attentions = [], []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(resnet_in_channels + res_skip_channels, out_channels, temb_channels,
                                         eps=resnet_eps, groups=resnet_groups, non_linearity=resnet_act_fn,
                                         operations=operations))
            attentions.append(Transformer2DModel(
                num_attention_heads, out_channels // num_attention_heads, in_channels=out_channels,
                num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups,
                use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention,
                operations=operations))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels,
                                                        operations=operations)])
        else:
            self.upsamplers = None
        self.resolution_idx = resolution_idx

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, encoder_hidden_states=None,
                upsample_size=None, return_res_samples=False, up_block_add_samples=None):
        output_states = () if return_res_samples else None
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            if return_res_samples:
                output_states = output_states + (hidden_states,)
            if up_block_add_samples is not None:
                hidden_states = hidden_states + up_block_add_samples.pop(0)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
            if return_res_samples:
                output_states = output_states + (hidden_states,)
            if up_block_add_samples is not None:
                hidden_states = hidden_states + up_block_add_samples.pop(0)
        if return_res_samples:
            return hidden_states, output_states
        return hidden_states


# ---------------------------------------------------------------------------
# BrushNet
# ---------------------------------------------------------------------------

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class BrushNetModel(nn.Module):
    def __init__(self, in_channels=4, conditioning_channels=5, flip_sin_to_cos=True, freq_shift=0,
                 down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
                 mid_block_type="UNetMidBlock2DCrossAttn",
                 up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
                 only_cross_attention=False, block_out_channels=(320, 640, 1280, 1280),
                 layers_per_block=2, downsample_padding=1, mid_block_scale_factor=1.0, act_fn="silu",
                 norm_num_groups=32, norm_eps=1e-5, cross_attention_dim=1280, transformer_layers_per_block=1,
                 attention_head_dim=8, num_attention_heads=None, use_linear_projection=False,
                 upcast_attention=False, resnet_time_scale_shift="default",
                 brushnet_conditioning_channel_order="rgb",
                 conditioning_embedding_out_channels=(16, 32, 96, 256), global_pool_conditions=False,
                 operations=comfy.ops.manual_cast):
        super().__init__()
        num_attention_heads = num_attention_heads or attention_head_dim
        self.config = dict(brushnet_conditioning_channel_order=brushnet_conditioning_channel_order,
                           global_pool_conditions=global_pool_conditions)

        conv_in_padding = (3 - 1) // 2
        self.conv_in_condition = operations.Conv2d(in_channels + conditioning_channels, block_out_channels[0],
                                                   kernel_size=3, padding=conv_in_padding)
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn, operations=operations)

        self.down_blocks = nn.ModuleList([])
        self.brushnet_down_blocks = nn.ModuleList([])
        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        output_channel = block_out_channels[0]
        self.brushnet_down_blocks.append(zero_module(
            operations.Conv2d(output_channel, output_channel, kernel_size=1)))
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            if down_block_type == "CrossAttnDownBlock2D":
                down_block = CrossAttnDownBlock2D(
                    num_layers=layers_per_block, in_channels=input_channel, out_channels=output_channel,
                    temb_channels=time_embed_dim, add_downsample=not is_final_block, resnet_eps=norm_eps,
                    resnet_act_fn=act_fn, resnet_groups=norm_num_groups, cross_attention_dim=cross_attention_dim,
                    num_attention_heads=num_attention_heads[i], downsample_padding=downsample_padding,
                    use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention[i],
                    operations=operations)
            elif down_block_type == "DownBlock2D":
                down_block = DownBlock2D(
                    num_layers=layers_per_block, in_channels=input_channel, out_channels=output_channel,
                    temb_channels=time_embed_dim, add_downsample=not is_final_block, resnet_eps=norm_eps,
                    resnet_act_fn=act_fn, resnet_groups=norm_num_groups, downsample_padding=downsample_padding,
                    operations=operations)
            else:
                raise ValueError(down_block_type)
            self.down_blocks.append(down_block)
            for _ in range(layers_per_block):
                self.brushnet_down_blocks.append(zero_module(
                    operations.Conv2d(output_channel, output_channel, kernel_size=1)))
            if not is_final_block:
                self.brushnet_down_blocks.append(zero_module(
                    operations.Conv2d(output_channel, output_channel, kernel_size=1)))

        mid_block_channel = block_out_channels[-1]
        self.brushnet_mid_block = zero_module(operations.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1))
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=mid_block_channel, temb_channels=time_embed_dim,
            num_layers=transformer_layers_per_block[-1], resnet_eps=norm_eps, resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups, cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads[-1], operations=operations)

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]

        self.up_blocks = nn.ModuleList([])
        self.brushnet_up_blocks = nn.ModuleList([])
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            add_upsample = not is_final_block
            if up_block_type == "CrossAttnUpBlock2D":
                up_block = CrossAttnUpBlock2D(
                    num_layers=layers_per_block + 1, in_channels=input_channel, out_channels=output_channel,
                    prev_output_channel=prev_output_channel, temb_channels=time_embed_dim,
                    add_upsample=add_upsample, resnet_eps=norm_eps, resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups, cross_attention_dim=cross_attention_dim,
                    num_attention_heads=reversed_num_attention_heads[i],
                    use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention[i],
                    resolution_idx=i, operations=operations)
            elif up_block_type == "UpBlock2D":
                up_block = UpBlock2D(
                    num_layers=layers_per_block + 1, in_channels=input_channel, out_channels=output_channel,
                    prev_output_channel=prev_output_channel, temb_channels=time_embed_dim,
                    add_upsample=add_upsample, resnet_eps=norm_eps, resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups, resolution_idx=i, operations=operations)
            else:
                raise ValueError(up_block_type)
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
            for _ in range(layers_per_block + 1):
                self.brushnet_up_blocks.append(zero_module(
                    operations.Conv2d(output_channel, output_channel, kernel_size=1)))
            if not is_final_block:
                self.brushnet_up_blocks.append(zero_module(
                    operations.Conv2d(output_channel, output_channel, kernel_size=1)))

    def forward(self, sample, timestep, encoder_hidden_states, brushnet_cond, conditioning_scale=1.0):
        if self.config["brushnet_conditioning_channel_order"] == "bgr":
            brushnet_cond = torch.flip(brushnet_cond, dims=[1])

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.int64, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        brushnet_cond = torch.concat([sample, brushnet_cond], 1)
        sample = self.conv_in_condition(brushnet_cond)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb,
                                                       encoder_hidden_states=encoder_hidden_states)
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        brushnet_down_block_res_samples = ()
        for down_block_res_sample, brushnet_down_block in zip(down_block_res_samples, self.brushnet_down_blocks):
            down_block_res_sample = brushnet_down_block(down_block_res_sample)
            brushnet_down_block_res_samples = brushnet_down_block_res_samples + (down_block_res_sample,)

        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
        brushnet_mid_block_res_sample = self.brushnet_mid_block(sample)

        up_block_res_samples = ()
        down_block_res_samples = list(down_block_res_samples)
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            upsample_size = None
            if not is_final_block:
                upsample_size = down_block_res_samples[-1].shape[2:]
            if getattr(upsample_block, "has_cross_attention", False):
                sample, up_res_samples = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=tuple(res_samples),
                    encoder_hidden_states=encoder_hidden_states, upsample_size=upsample_size,
                    return_res_samples=True)
            else:
                sample, up_res_samples = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=tuple(res_samples),
                    upsample_size=upsample_size, return_res_samples=True)
            up_block_res_samples += up_res_samples

        brushnet_up_block_res_samples = ()
        for up_block_res_sample, brushnet_up_block in zip(up_block_res_samples, self.brushnet_up_blocks):
            up_block_res_sample = brushnet_up_block(up_block_res_sample)
            brushnet_up_block_res_samples = brushnet_up_block_res_samples + (up_block_res_sample,)

        brushnet_down_block_res_samples = [s * conditioning_scale for s in brushnet_down_block_res_samples]
        brushnet_mid_block_res_sample = brushnet_mid_block_res_sample * conditioning_scale
        brushnet_up_block_res_samples = [s * conditioning_scale for s in brushnet_up_block_res_samples]

        return (brushnet_down_block_res_samples, brushnet_mid_block_res_sample, brushnet_up_block_res_samples)


# ---------------------------------------------------------------------------
# UNetMotionModel
# ---------------------------------------------------------------------------

class UNetMotionModel(nn.Module):
    def __init__(self, sample_size=None, in_channels=4, conditioning_channels=3, out_channels=4,
                 down_block_types=("CrossAttnDownBlockMotion", "CrossAttnDownBlockMotion",
                                   "CrossAttnDownBlockMotion", "DownBlockMotion"),
                 mid_block_type="UNetMidBlockCrossAttnMotion",
                 up_block_types=("UpBlockMotion", "CrossAttnUpBlockMotion", "CrossAttnUpBlockMotion",
                                 "CrossAttnUpBlockMotion"),
                 block_out_channels=(320, 640, 1280, 1280), layers_per_block=2, downsample_padding=1,
                 mid_block_scale_factor=1.0, act_fn="silu", norm_num_groups=32, norm_eps=1e-5,
                 cross_attention_dim=1280, use_linear_projection=False, num_attention_heads=8,
                 motion_max_seq_length=32, motion_num_attention_heads=8, use_motion_mid_block=True,
                 operations=comfy.ops.manual_cast):
        super().__init__()
        self.sample_size = sample_size
        self.config = dict(in_channels=in_channels)

        self.conv_in = operations.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], True, 0)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim, act_fn=act_fn,
                                                operations=operations)
        self.encoder_hid_proj = None

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            if down_block_type == "CrossAttnDownBlockMotion":
                down_block = CrossAttnDownBlockMotion(
                    num_layers=layers_per_block, in_channels=input_channel, out_channels=output_channel,
                    temb_channels=time_embed_dim, add_downsample=not is_final_block, resnet_eps=norm_eps,
                    resnet_act_fn=act_fn, resnet_groups=norm_num_groups, cross_attention_dim=cross_attention_dim,
                    num_attention_heads=num_attention_heads[i], downsample_padding=downsample_padding,
                    use_linear_projection=use_linear_projection,
                    temporal_num_attention_heads=motion_num_attention_heads,
                    temporal_max_seq_length=motion_max_seq_length, operations=operations)
            elif down_block_type == "DownBlockMotion":
                down_block = DownBlockMotion(
                    num_layers=layers_per_block, in_channels=input_channel, out_channels=output_channel,
                    temb_channels=time_embed_dim, add_downsample=not is_final_block, resnet_eps=norm_eps,
                    resnet_act_fn=act_fn, resnet_groups=norm_num_groups, downsample_padding=downsample_padding,
                    temporal_num_attention_heads=motion_num_attention_heads,
                    temporal_max_seq_length=motion_max_seq_length, operations=operations)
            else:
                raise ValueError(down_block_type)
            self.down_blocks.append(down_block)

        # vendored UNetMotionModel ignores mid_block_type and selects on
        # use_motion_mid_block only (the DiffuEraser config sets
        # mid_block_type="UNetMidBlock2DCrossAttn" yet ships motion modules)
        if use_motion_mid_block:
            self.mid_block = UNetMidBlockCrossAttnMotion(
                in_channels=block_out_channels[-1], temb_channels=time_embed_dim, resnet_eps=norm_eps,
                resnet_act_fn=act_fn, resnet_groups=norm_num_groups, cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[-1],
                temporal_num_attention_heads=motion_num_attention_heads,
                temporal_max_seq_length=motion_max_seq_length, operations=operations)
        else:
            self.mid_block = UNetMidBlock2DCrossAttn(
                in_channels=block_out_channels[-1], temb_channels=time_embed_dim, resnet_eps=norm_eps,
                resnet_act_fn=act_fn, resnet_groups=norm_num_groups, cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[-1], operations=operations)

        self.num_upsamplers = 0
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            add_upsample = not is_final_block
            if add_upsample:
                self.num_upsamplers += 1
            if up_block_type == "CrossAttnUpBlockMotion":
                up_block = CrossAttnUpBlockMotion(
                    num_layers=layers_per_block + 1, in_channels=input_channel, out_channels=output_channel,
                    prev_output_channel=prev_output_channel, temb_channels=time_embed_dim,
                    add_upsample=add_upsample, resnet_eps=norm_eps, resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups, cross_attention_dim=cross_attention_dim,
                    num_attention_heads=reversed_num_attention_heads[i], use_linear_projection=use_linear_projection,
                    resolution_idx=i,
                    temporal_num_attention_heads=motion_num_attention_heads,
                    temporal_max_seq_length=motion_max_seq_length, operations=operations)
            elif up_block_type == "UpBlockMotion":
                up_block = UpBlockMotion(
                    num_layers=layers_per_block + 1, in_channels=input_channel, out_channels=output_channel,
                    prev_output_channel=prev_output_channel, temb_channels=time_embed_dim,
                    add_upsample=add_upsample, resnet_eps=norm_eps, resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups, resolution_idx=i,
                    temporal_num_attention_heads=motion_num_attention_heads,
                    temporal_max_seq_length=motion_max_seq_length, operations=operations)
            else:
                raise ValueError(up_block_type)
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = operations.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups,
                                                  eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = operations.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, sample, timestep, encoder_hidden_states, num_frames=24,
                down_block_add_samples=None, mid_block_add_sample=None, up_block_add_samples=None):
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.int64, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0] // num_frames)

        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        emb = emb.repeat_interleave(repeats=num_frames, dim=0)

        encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=num_frames, dim=0)

        is_brushnet = (down_block_add_samples is not None and mid_block_add_sample is not None
                       and up_block_add_samples is not None)

        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        if is_brushnet:
            down_block_add_samples = list(down_block_add_samples)
            up_block_add_samples = list(up_block_add_samples)
            sample = sample + down_block_add_samples.pop(0)

        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                if is_brushnet and len(down_block_add_samples) > 0:
                    additional_residuals["down_block_add_samples"] = [
                        down_block_add_samples.pop(0)
                        for _ in range(len(downsample_block.resnets) + (downsample_block.downsamplers is not None))]
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb,
                                                       encoder_hidden_states=encoder_hidden_states,
                                                       num_frames=num_frames, **additional_residuals)
            else:
                additional_residuals = {}
                if is_brushnet and len(down_block_add_samples) > 0:
                    additional_residuals["down_block_add_samples"] = [
                        down_block_add_samples.pop(0)
                        for _ in range(len(downsample_block.resnets) + (downsample_block.downsamplers is not None))]
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, num_frames=num_frames,
                                                       **additional_residuals)
            down_block_res_samples += res_samples

        if self.mid_block is not None:
            sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states,
                                    num_frames=num_frames, mid_block_add_sample=mid_block_add_sample)

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            additional_residuals = {}
            if is_brushnet and len(up_block_add_samples) > 0:
                additional_residuals["up_block_add_samples"] = [
                    up_block_add_samples.pop(0)
                    for _ in range(len(upsample_block.resnets) + (upsample_block.upsamplers is not None))]
            if getattr(upsample_block, "has_cross_attention", False):
                sample = upsample_block(hidden_states=sample, temb=emb,
                                        res_hidden_states_tuple=res_samples,
                                        encoder_hidden_states=encoder_hidden_states,
                                        upsample_size=upsample_size, num_frames=num_frames, **additional_residuals)
            else:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples,
                                        upsample_size=upsample_size, num_frames=num_frames, **additional_residuals)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return (sample,)


def build_unet_motion_from_config(config, operations=comfy.ops.manual_cast):
    cfg = dict(config)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    known = {
        "sample_size", "in_channels", "conditioning_channels", "out_channels", "down_block_types",
        "mid_block_type", "up_block_types", "block_out_channels", "layers_per_block", "downsample_padding",
        "mid_block_scale_factor", "act_fn", "norm_num_groups", "norm_eps", "cross_attention_dim",
        "use_linear_projection", "num_attention_heads", "motion_max_seq_length", "motion_num_attention_heads",
        "use_motion_mid_block",
    }
    kwargs = {k: v for k, v in cfg.items() if k in known}
    if "num_attention_heads" not in kwargs and "attention_head_dim" in cfg:
        kwargs["num_attention_heads"] = cfg["attention_head_dim"]
    if kwargs.get("num_attention_heads") is None:
        kwargs["num_attention_heads"] = kwargs.get("attention_head_dim", 8)
    return UNetMotionModel(operations=operations, **kwargs)


def build_brushnet_from_config(config, operations=comfy.ops.manual_cast):
    cfg = dict(config)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    known = {
        "in_channels", "conditioning_channels", "flip_sin_to_cos", "freq_shift", "down_block_types",
        "mid_block_type", "up_block_types", "only_cross_attention", "block_out_channels", "layers_per_block",
        "downsample_padding", "mid_block_scale_factor", "act_fn", "norm_num_groups", "norm_eps",
        "cross_attention_dim", "transformer_layers_per_block", "attention_head_dim", "num_attention_heads",
        "use_linear_projection", "upcast_attention", "resnet_time_scale_shift",
        "brushnet_conditioning_channel_order", "conditioning_embedding_out_channels", "global_pool_conditions",
    }
    kwargs = {k: v for k, v in cfg.items() if k in known}
    return BrushNetModel(operations=operations, **kwargs)


# ---------------------------------------------------------------------------
# TCD scheduler (port of diffusers TCDScheduler, epsilon / default config)
# ---------------------------------------------------------------------------

class TCDScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                 beta_schedule="scaled_linear", original_inference_steps=50, set_alpha_to_one=True):
        if beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        else:
            raise NotImplementedError(beta_schedule)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        self.num_train_timesteps = num_train_timesteps
        self.original_inference_steps = original_inference_steps
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    def set_timesteps(self, num_inference_steps, device=None):
        original_steps = self.original_inference_steps
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError("num_inference_steps too large")
        k = self.num_train_timesteps // original_steps
        tcd_origin_timesteps = np.asarray(list(range(1, int(original_steps) + 1))) * k - 1
        tcd_origin_timesteps = tcd_origin_timesteps[::-1].copy()
        inference_indices = np.linspace(0, len(tcd_origin_timesteps), num=num_inference_steps, endpoint=False)
        inference_indices = np.floor(inference_indices).astype(np.int64)
        timesteps = tcd_origin_timesteps[inference_indices]
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.long)

    def _index_for_timestep(self, timestep):
        schedule = self.timesteps
        indices = (schedule == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def step(self, model_output, timestep, sample, eta=0.0, generator=None, prediction_type="epsilon"):
        # stateless per-timestep step: reproduces the reset-per-context scheduler
        # behaviour of the original pipeline (scheduler __dict__ restored each call)
        step_index = self._index_for_timestep(timestep)
        assert 0 <= eta <= 1.0

        prev_step_index = step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = torch.tensor(0)

        timestep_s = torch.floor((1 - eta) * prev_timestep).to(dtype=torch.long)

        alpha_prod_t = self.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        alpha_prod_s = self.alphas_cumprod[timestep_s]
        beta_prod_s = 1 - alpha_prod_s

        if prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
            pred_epsilon = model_output
        elif prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
            pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
        elif prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        else:
            raise ValueError(prediction_type)
        pred_noised_sample = alpha_prod_s.sqrt() * pred_original_sample + beta_prod_s.sqrt() * pred_epsilon

        if eta > 0 and step_index != self.num_inference_steps - 1:
            noise = torch.randn(model_output.shape, generator=generator, device=model_output.device,
                                dtype=pred_noised_sample.dtype)
            prev_sample = (alpha_prod_t_prev / alpha_prod_s).sqrt() * pred_noised_sample + (
                1 - alpha_prod_t_prev / alpha_prod_s).sqrt() * noise
        else:
            prev_sample = pred_noised_sample
        return prev_sample

    def add_noise(self, original_samples, noise, timesteps):
        # DDPM-style forward noise (used by UniPC-config scheduler in the original)
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise


class UniPCAddNoiseScheduler:
    """Replicates diffusers UniPCMultistepScheduler.add_noise in its default
    (set_timesteps never called) state, which is what the original DiffuEraser
    pipeline actually used for priori injection.

    Quirk preserved: timesteps table is [999..0] (float32), sigmas[i] indexes
    alphas_cumprod[i]; looking up timestep 0 lands on table position 999, i.e.
    the *noisiest* sigma — priori latents are injected almost as pure noise
    (alpha≈0.068, sigma≈0.998). This is the behaviour the checkpoint produces
    its documented results with."""

    def __init__(self, scheduler_config):
        cfg = dict(scheduler_config)
        n = cfg["num_train_timesteps"]
        if cfg.get("beta_schedule") == "scaled_linear":
            betas = torch.linspace(cfg["beta_start"] ** 0.5, cfg["beta_end"] ** 0.5, n, dtype=torch.float32) ** 2
        else:
            betas = torch.linspace(cfg["beta_start"], cfg["beta_end"], n, dtype=torch.float32)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        timesteps = np.linspace(0, n - 1, n, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)

    def _index_for_timestep(self, t, schedule_timesteps):
        indices = (schedule_timesteps == t).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def add_noise(self, original_samples, noise, timesteps):
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)
        step_indices = [self._index_for_timestep(t, schedule_timesteps) for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        alpha_t = 1 / ((sigma ** 2 + 1) ** 0.5)
        sigma_t = sigma * alpha_t
        return alpha_t * original_samples + sigma_t * noise


def build_add_noise_scheduler(scheduler_config):
    # replicates noise_scheduler = UniPC(config from sd15_repo scheduler folder)
    return UniPCAddNoiseScheduler(scheduler_config)


# ---------------------------------------------------------------------------
# PCM LoRA merging (kohya-format keys: lora_unet_<module_path>.lora_down/up/alpha)
# ---------------------------------------------------------------------------

def merge_pcm_lora(model, lora_path, strength=1.0, device=None):
    from safetensors.torch import load_file
    lora_sd = load_file(lora_path, device="cpu")
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())

    merged = 0
    skipped = []
    # group keys by target module
    modules = {k[:-len(".lora_down.weight")]: k for k in lora_sd.keys() if k.endswith(".lora_down.weight")}
    with torch.no_grad():
        for mkey, down_key in modules.items():
            # kohya name: "lora_unet_" + module_key.replace(".", "_");
            # module_key is the param key without the trailing ".weight"
            target = None
            candidates = [mk for mk in model_keys
                          if mk.endswith(".weight")
                          and "lora_unet_" + mk[:-len(".weight")].replace(".", "_") == mkey]
            if len(candidates) == 1:
                target = candidates[0]
            if target is None:
                skipped.append(mkey)
                continue
            up_key = mkey + ".lora_up.weight"
            alpha_key = mkey + ".alpha"
            down = lora_sd[down_key].float()
            up = lora_sd[up_key].float()
            rank = down.shape[0] if down.ndim >= 2 else down.shape[-1]
            if alpha_key in lora_sd:
                scale = lora_sd[alpha_key].float().item() / rank
            else:
                scale = 1.0 / rank
            w = model_sd[target]
            if down.ndim == 4:  # conv kernel
                out_c = up.shape[0]
                r = up.shape[1]
                down_m = down.reshape(r, -1)
                up_m = up.reshape(out_c, r)
                w_delta = (up_m @ down_m).reshape(w.shape)
            else:
                w_delta = up @ down
            w_new = (w.float() + (strength * scale * w_delta).to(device=w.device, dtype=torch.float32)).to(w.dtype)
            _apply_weight(model, target, w_new)
            merged += 1
    return merged, skipped


def _apply_weight(model, key, value):
    # splits "a.b.0.c.weight" style paths and assigns the tensor as a Parameter
    parts = key.split(".")
    obj = model
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    last = parts[-1]
    setattr(obj, last, torch.nn.Parameter(value, requires_grad=False))


def load_lora_for_unet(unet, lora_path, strength=1.0):
    if lora_path is None:
        return 0, []
    return merge_pcm_lora(unet, lora_path, strength=strength)


# ---------------------------------------------------------------------------
# image helpers (VaeImageProcessor-compatible preprocessing)
# ---------------------------------------------------------------------------

def preprocess_image_batch(pil_images, height, width, device, dtype):
    # VaeImageProcessor.preprocess: RGB convert, resize, [0,1] -> [-1,1]
    from PIL import Image
    tensors = []
    for img in pil_images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height))
        arr = torch.from_numpy(np.array(img)).float() / 255.0
        arr = arr.permute(2, 0, 1)  # c h w
        tensors.append(arr)
    x = torch.stack(tensors)  # f c h w
    x = 2.0 * x - 1.0
    return x.to(device=device, dtype=dtype)


def postprocess_to_pil(pixels):
    # pixels: (f, h, w, c) float in [0,1]
    from PIL import Image
    frames = []
    for i in range(pixels.shape[0]):
        arr = (pixels[i].detach().float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        frames.append(Image.fromarray(arr))
    return frames


# ---------------------------------------------------------------------------
# context windows (verbatim port)
# ---------------------------------------------------------------------------

def get_frames_context_swap(total_frames=192, overlap=4, num_frames_per_clip=24):
    if total_frames < num_frames_per_clip:
        num_frames_per_clip = total_frames
    context_list = []
    context_list_swap = []
    for i in range(1, 2):
        sample_interval = np.array(range(0, total_frames, i))
        n = len(sample_interval)
        if n > num_frames_per_clip:
            for k in range(0, n - num_frames_per_clip, num_frames_per_clip - overlap):
                context_list.append(sample_interval[k:k + num_frames_per_clip])
            if k + num_frames_per_clip < n and i == 1:
                context_list.append(sample_interval[n - num_frames_per_clip:n])
            context_list_swap.append(sample_interval[0:num_frames_per_clip])
            for k in range(num_frames_per_clip // 2, n - num_frames_per_clip, num_frames_per_clip - overlap):
                context_list_swap.append(sample_interval[k:k + num_frames_per_clip])
            if k + num_frames_per_clip < n and i == 1:
                context_list_swap.append(sample_interval[n - num_frames_per_clip:n])
        if n == num_frames_per_clip:
            context_list.append(sample_interval[n - num_frames_per_clip:n])
            context_list_swap.append(sample_interval[n - num_frames_per_clip:n])
    return context_list, context_list_swap


# ---------------------------------------------------------------------------
# model loading + inference orchestration
# ---------------------------------------------------------------------------

class DiffuEraserModel:
    """Holds the comfy patchers, the comfy VAE and the schedulers; runs inference."""

    def __init__(self, unet, brushnet, vae, noise_scheduler, device=None, dtype=torch.float16):
        self.unet = unet
        self.brushnet = brushnet
        self.vae = vae  # comfy.sd.VAE
        self.noise_scheduler = noise_scheduler
        self.dtype = dtype
        self.device = device or mm.get_torch_device()

        load_device = mm.get_torch_device()
        self.unet_patcher = comfy.model_patcher.ModelPatcher(unet, load_device=load_device, offload_device=torch.device("cpu"))
        self.brushnet_patcher = comfy.model_patcher.ModelPatcher(brushnet, load_device=load_device, offload_device=torch.device("cpu"))
        self._loaded = False

    def load_for_inference(self):
        # comfy model_management decides placement/offload (replaces the old
        # xformers + group-offloading setup). Models are small (~3.4GB fp16 +
        # 0.16GB VAE); full load keeps everything resident for the whole run —
        # re-issuing load_models_gpu per VAE batch would re-stage the VAE
        # ("prepared for dynamic VRAM loading") on every call.
        if not self._loaded:
            mm.load_models_gpu([self.unet_patcher, self.brushnet_patcher, self.vae.patcher],
                               force_full_load=True)
            self._loaded = True

    def unload(self):
        """Two-stage workflow: free the diffusion models after sampling."""
        mm.unload_model_and_clones(self.unet_patcher)
        mm.unload_model_and_clones(self.brushnet_patcher)
        self._loaded = False
        mm.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- encoders -----------------------------------------------------
    def encode_pixels(self, pixel_values):
        # pixel_values: (f, c, h, w) in [-1,1] -> scaled latents (f,4,h/8,w/8)
        # replicates diffusers vae.encode(...).latent_dist.sample()
        # NOTE: comfy's Encoder (like diffusers') expects [-1,1]; the [0,1]-to-
        # [-1,1] conversion lives in comfy VAE.encode's process_input, which we
        # bypass by calling first_stage_model directly. The VAE patcher is
        # loaded once in load_for_inference — do NOT re-issue load_models_gpu
        # here (it re-stages the VAE on every call).
        pixels = pixel_values.to(device=self.vae.device, dtype=self.vae.vae_dtype)
        fs = self.vae.first_stage_model
        with torch.no_grad():
            z = fs.encoder(pixels)
            z = fs.quant_conv(z)
            mean, logvar = z.chunk(2, dim=1)
            std = torch.exp(0.5 * logvar)
            latents = mean + std * torch.randn_like(std)
        return latents.to(device=self.device, dtype=self.dtype) * VAE_SCALE_FACTOR

    def decode_latents(self, latents):
        raw = (1.0 / VAE_SCALE_FACTOR) * latents
        raw = raw.float()
        pixels = self.vae.decode(raw)  # (f, h, w, c) in [0,1]
        return pixels

    # ---- model forward -------------------------------------------------
    def model_forward(self, latents_j, t, positive_embeds, brushnet_cond, guidance_scale,
                      negative_embeds=None):
        """One (possibly CFG) forward over a context window.
        latents_j: (f,4,h,w); positive_embeds: (1,77,768); brushnet_cond: (f,5,h,w)"""
        do_cfg = guidance_scale > 1.0
        num_frames = latents_j.shape[0]
        pos = positive_embeds.to(device=self.device, dtype=self.dtype)
        if do_cfg:
            if negative_embeds is None:
                raise ValueError("negative conditioning required when guidance_scale > 1")
            neg = negative_embeds.to(device=self.device, dtype=self.dtype)
            cond = torch.cat([neg, pos])          # (2,77,768): uncond first, like diffusers
            latent_model_input = torch.cat([latents_j] * 2)
            b = 2
        else:
            cond = pos
            latent_model_input = latents_j
            b = 1

        brushnet_embeds = cond.repeat_interleave(num_frames, dim=0)          # (b*f,77,768)
        brushnet_cond_in = torch.cat([brushnet_cond] * b) if do_cfg else brushnet_cond

        down_res, mid_res, up_res = self.brushnet(
            latent_model_input, t, brushnet_embeds, brushnet_cond_in, conditioning_scale=1.0)
        noise_pred = self.unet(latent_model_input, t, cond, num_frames=num_frames,
                               down_block_add_samples=down_res, mid_block_add_sample=mid_res,
                               up_block_add_samples=up_res)[0]

        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred

    def pipeline_loop(self, latents, brushnet_cond_all, positive_embeds, num_frames, num_inference_steps,
                      guidance_scale, generator, video_length, eta=0.0, negative_embeds=None, callback=None):
        tcd = TCDScheduler()
        tcd.set_timesteps(num_inference_steps, device=self.device)
        timesteps = tcd.timesteps
        latents = latents.to(device=self.device, dtype=self.dtype)

        overlap = num_frames // 4
        context_list, context_list_swap = get_frames_context_swap(video_length, overlap=overlap,
                                                                  num_frames_per_clip=num_frames)
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

        for i, t in enumerate(timesteps):
            count.zero_()
            value.zero_()
            if (i % 2 == 1):
                context_list_choose = context_list_swap
            else:
                context_list_choose = context_list

            for j, context in enumerate(context_list_choose):
                latents_j = latents[context, :, :, :]
                noise_pred = self.model_forward(
                    latents_j, t, positive_embeds, brushnet_cond_all[context, :, :, :],
                    guidance_scale=guidance_scale, negative_embeds=negative_embeds)
                latents_j = tcd.step(noise_pred, t, latents_j, eta=eta, generator=generator)
                count[context, ...] += 1
                if j == 0:
                    value[context, ...] += latents_j
                else:
                    overlap_index_list = [index for index, cval in enumerate(count[context, 0, 0, 0]) if cval > 1]
                    if overlap_index_list:
                        overlap_cur = len(overlap_index_list)
                        ratio_next = torch.linspace(0, 1, overlap_cur + 2)[1:-1].to(latents.device)
                        ratio_pre = 1 - ratio_next
                        for i_overlap in overlap_index_list:
                            value[context[i_overlap], ...] = value[context[i_overlap], ...] * ratio_pre[i_overlap] + latents_j[i_overlap, ...] * ratio_next[i_overlap]
                        # quirk preserved: original uses the loop-leaked last index
                        i_overlap = overlap_index_list[-1]
                        value[context[i_overlap:num_frames], ...] = latents_j[i_overlap:num_frames, ...]

            latents = value.clone()
            if callback is not None:
                callback(i, t, latents)
        return latents

    # ---- brushnet conditioning ------------------------------------------
    def build_brushnet_cond(self, masked_images, masks):
        """masked_images/masks: PIL lists -> (F,5,h/8,w/8) conditioning tensor.
        Replicates prepare_image + (sum(1)<0) trick + latent concat."""
        h, w = masked_images[0].size[1], masked_images[0].size[0]
        imgs = preprocess_image_batch(masked_images, h, w, self.device, self.dtype)      # (F,C,H,W)
        mks = preprocess_image_batch(masks, h, w, self.device, self.dtype)               # (F,C,H,W)
        binary = (mks.sum(1)[:, None, :, :] < 0).to(imgs.dtype)
        binary = F.interpolate(binary, size=(imgs.shape[-2] // 8, imgs.shape[-1] // 8))
        cond_latents = []
        for i in range(0, imgs.shape[0], 16):
            cond_latents.append(self.encode_pixels(imgs[i:i + 16]))
        cond_latents = torch.cat(cond_latents, dim=0)
        return torch.concat([cond_latents, binary], 1)

    # ---- full DiffuEraser inference (port of DiffuEraser.forward) -------
    def run_inference(self, frames, masks_input, prioris, positive_embeds, negative_embeds=None,
                      num_frames=22, num_inference_steps=2, guidance_scale=0.0, seed=None,
                      mask_dilation_iter=4, blended=False, if_save_video=False, output_path=None,
                      fps=24, eta=0.0, callback=None, dump_dir=None):
        import cv2
        import copy
        import random
        from PIL import Image
        from einops import repeat as einops_repeat

        def dump(name, tensor):
            if dump_dir is not None:
                os.makedirs(dump_dir, exist_ok=True)
                torch.save(tensor.detach().float().cpu(), os.path.join(dump_dir, name + ".pt"))

        self.load_for_inference()

        frames = list(frames)
        masks_input = list(masks_input)
        prioris = list(prioris)
        tar_width, tar_height = frames[0].size
        img_size = (tar_width, tar_height)
        real_video_length = len(frames)

        if real_video_length < 22:
            raise ValueError("The effective video duration is too short (need at least 22 frames).")

        # ---- masks: binarize + erode/dilate + masked images (verbatim) ----
        validation_masks_input = []
        validation_images_input = []
        for idx, mask in enumerate(masks_input):
            mask = mask.convert('L')
            if mask.size != img_size:
                mask = mask.resize(img_size, Image.NEAREST)
            mask = np.asarray(mask)
            m = np.array(mask > 0).astype(np.uint8)
            m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=mask_dilation_iter)
            mask_img = Image.fromarray(m * 255)
            validation_masks_input.append(mask_img)
            masked_image = np.array(frames[idx]) * (1 - (np.array(mask_img)[:, :, np.newaxis].astype(np.float32) / 255))
            validation_images_input.append(Image.fromarray(masked_image.astype(np.uint8)))

        if prioris[0].size != img_size:
            prioris = [img.resize(img_size) for img in prioris]

        n_total_frames = min(min(len(frames), len(validation_masks_input)), len(prioris))
        validation_masks_input = validation_masks_input[:n_total_frames]
        validation_images_input = validation_images_input[:n_total_frames]
        frames = frames[:n_total_frames]
        prioris = prioris[:n_total_frames]

        validation_masks_input_ori = copy.deepcopy(validation_masks_input)
        resized_frames_ori = copy.deepcopy(frames)

        # ---- noise --------------------------------------------------------
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        shape = (num_frames, 4, tar_height // 8, tar_width // 8)
        noise_pre = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        dump("noise_pre", noise_pre)
        n_clip = int(np.ceil(n_total_frames / num_frames))
        noise = einops_repeat(noise_pre, "t c h w -> (repeat t) c h w", repeat=n_clip)[:real_video_length, ...]

        # ---- encode prioris ------------------------------------------------
        pixel_values = []
        for image in prioris:
            pixel_values.append(preprocess_image_batch([image], tar_height, tar_width, self.device, torch.float32)[0])
        pixel_values = torch.stack(pixel_values).to(dtype=self.dtype)
        latents = []
        for i in range(0, pixel_values.shape[0], 16):
            latents.append(self.encode_pixels(pixel_values[i:i + 16]))
        latents = torch.cat(latents, dim=0)
        dump("priori_latents", latents)

        timesteps = torch.tensor([0], device=self.device).long()

        # ---- pre-inference --------------------------------------------------
        if n_total_frames > num_frames * 2:
            step = n_total_frames / num_frames
            sample_index = [int(i * step) for i in range(num_frames)]
            sample_index = sample_index[:num_frames]  # original hardcoded 22 (= default nframes)
            validation_masks_input_pre = [validation_masks_input[i] for i in sample_index]
            validation_images_input_pre = [validation_images_input[i] for i in sample_index]
            latents_pre = torch.stack([latents[i] for i in sample_index])

            latents_pre = self.noise_scheduler.add_noise(latents_pre, noise_pre, timesteps)

            cond_pre = self.build_brushnet_cond(validation_images_input_pre, validation_masks_input_pre)
            latents_pre_out = self.pipeline_loop(
                latents_pre, cond_pre, positive_embeds, num_frames, num_inference_steps,
                guidance_scale, generator, video_length=num_frames, eta=eta,
                negative_embeds=negative_embeds, callback=callback)

            images_pre_px = self.decode_latents(latents_pre_out.float())
            images_pre_out = postprocess_to_pil(images_pre_px)

            # replace input frames with updated frames
            black_image = Image.new('L', validation_masks_input[0].size, color=0)
            for i, index in enumerate(sample_index):
                latents[index] = latents_pre_out[i]
                validation_masks_input[index] = black_image
                validation_images_input[index] = images_pre_out[i]
                frames[index] = images_pre_out[i]

        # ---- frame-by-frame inference ---------------------------------------
        latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        dump("latents_main_in", latents)
        cond_all = self.build_brushnet_cond(validation_images_input, validation_masks_input)
        dump("cond_all", cond_all)

        out_latents = self.pipeline_loop(
            latents, cond_all, positive_embeds, num_frames, num_inference_steps,
            guidance_scale, generator, video_length=n_total_frames, eta=eta,
            negative_embeds=negative_embeds, callback=callback)

        images_px = self.decode_latents(out_latents.float())
        dump("out_latents", out_latents)
        out_frames = postprocess_to_pil(images_px)
        images = out_frames[:real_video_length]

        # ---- compose ----------------------------------------------------------
        binary_masks = validation_masks_input_ori
        if blended:
            mask_blurreds = []
            for i in range(len(binary_masks)):
                mask_array = np.array(binary_masks[i])
                mask_blurred = morphological_edge_blur(np.array(mask_array), sigma=2.0, edge_width=3)
                binary_mask = 1 - (1 - mask_array / 255.) * (1 - mask_blurred)
                mask_blurreds.append(Image.fromarray((binary_mask * 255).astype(np.uint8)))
            binary_masks = mask_blurreds

            comp_frames = []
            for i in range(len(images)):
                mask = np.expand_dims(np.array(binary_masks[i]), 2).repeat(3, axis=2).astype(np.float32) / 255.
                img = (np.array(images[i]).astype(np.uint8) * mask + np.array(resized_frames_ori[i]).astype(np.uint8) * (1 - mask)).astype(np.uint8)
                comp_frames.append(Image.fromarray(img))
        else:
            comp_frames = simple_flicker_smoothing(images, alpha=0.15)

        if if_save_video:
            prefix = ''.join(random.choice("0123456789") for _ in range(6))
            save_path = os.path.join(output_path or ".", f"diffueraser_{prefix}.mp4")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, comp_frames[0].size)
            for f in range(real_video_length):
                img = np.array(comp_frames[f]).astype(np.uint8)
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            writer.release()

        return comp_frames


def simple_flicker_smoothing(frames, alpha=0.1):
    """Simple flicker smoothing that minimizes impact on dynamic content."""
    if len(frames) < 2:
        return frames
    smoothed_frames = [frames[0]]
    for i in range(1, len(frames)):
        current = np.array(frames[i]).astype(np.float32)
        previous = np.array(frames[i - 1]).astype(np.float32)
        diff = np.abs(current - previous)
        static_mask = (diff < 10.0).astype(np.float32)
        smoothed = previous * alpha * static_mask + current * (1 - alpha * static_mask)
        smoothed_frames.append(Image.fromarray(np.clip(smoothed, 0, 255).astype(np.uint8)))
    return smoothed_frames


def morphological_edge_blur(mask, sigma=3.0, edge_width=5):
    """Edge-only blur of the mask, used for blended composition."""
    import cv2
    if mask.dtype != np.float32:
        mask_float = mask.astype(np.float32)
    else:
        mask_float = mask.copy()
    binary_mask = (mask_float > 0.5).astype(np.uint8)
    if not np.any(binary_mask):
        return mask_float
    kernel_size = max(3, edge_width)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    edge_mask = binary_mask - eroded
    edge_region = mask_float * edge_mask.astype(np.float32)
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    ksize = max(3, min(101, ksize if ksize % 2 == 1 else ksize + 1))
    blurred_edges = cv2.GaussianBlur(edge_region, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    inner_region = mask_float * eroded.astype(np.float32)
    result = inner_region + blurred_edges
    return np.clip(result, 0.0, 1.0)



def load_diffueraser_core(diffueraser_dir, vae_path, pcm_lora_path, repo_dir, scheduler_config_path,
                          device=None, dtype=torch.float16):
    """Loads BrushNet + UNetMotionModel from models/DiffuEraser and the comfy VAE."""
    operations = comfy.ops.manual_cast

    with open(os.path.join(diffueraser_dir, "unet_main", "config.json")) as f:
        unet_config = json.load(f)
    with open(os.path.join(diffueraser_dir, "brushnet", "config.json")) as f:
        brushnet_config = json.load(f)

    unet_sd = load_torch_file(os.path.join(diffueraser_dir, "unet_main", "diffusion_pytorch_model.safetensors"))
    brushnet_sd = load_torch_file(os.path.join(diffueraser_dir, "brushnet", "diffusion_pytorch_model.safetensors"))
    if dtype is not None:
        unet_sd = {k: v.to(dtype) for k, v in unet_sd.items()}
        brushnet_sd = {k: v.to(dtype) for k, v in brushnet_sd.items()}

    unet = build_unet_motion_from_config(unet_config, operations=operations)
    missing, unexpected = unet.load_state_dict(unet_sd, strict=True)
    unet.requires_grad_(False)
    if dtype is not None:
        unet.to(dtype)  # plain nn.LayerNorm params keep their fp32 init after load_state_dict

    brushnet = build_brushnet_from_config(brushnet_config, operations=operations)
    missing, unexpected = brushnet.load_state_dict(brushnet_sd, strict=True)
    brushnet.requires_grad_(False)
    if dtype is not None:
        brushnet.to(dtype)

    if pcm_lora_path:
        merged, skipped = load_lora_for_unet(unet, pcm_lora_path)
        print(f"[DiffuEraser] merged {merged} PCM LoRA modules ({len(skipped)} skipped)")

    vae_sd = load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=vae_sd)

    with open(scheduler_config_path) as f:
        scheduler_config = json.load(f)
    noise_scheduler = build_add_noise_scheduler(scheduler_config)

    return DiffuEraserModel(unet, brushnet, vae, noise_scheduler, device=device, dtype=dtype)
