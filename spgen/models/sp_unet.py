"""SDXL UNet with Layer-wise Self-Attention (LSA) for multi-layer SP map latents.

The latents of the ``num_layers`` SP maps of an object are flattened into the batch dimension,
i.e. ``(B * num_layers, 4, h, w)``. The self-attention processors of the UNet are replaced so that
the tokens of all SP layers of one object are concatenated and attend to each other (Sec. 3.2.2).
The input image conditions the model through cross-attention on its DINOv2 tokens.
"""
import math

import torch
from torch import nn
from einops import rearrange, repeat
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor
from diffusers.utils.import_utils import is_xformers_available


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding added to the condition tokens of each SP layer."""

    def __init__(self, d_model, max_len=10):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x, offset=0):
        """x: (b, n, l, c). The i-th entry along n receives the encoding of index i + offset."""
        return x + self.pe[:, offset:offset + x.shape[1]].unsqueeze(2).to(x.dtype)


class LayerwiseAttnProcessor:
    """Wraps a self-attention (``attn1``) processor: the tokens of all SP layers of an object are concatenated."""

    def __init__(self, chained_proc):
        self.chained_proc = chained_proc

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, num_layers=1):
        if num_layers > 1:
            hidden_states = rearrange(hidden_states, "(b f) l c -> b (f l) c", f=num_layers).contiguous()
        res = self.chained_proc(attn, hidden_states, None, attention_mask)
        if num_layers > 1:
            res = rearrange(res, "b (f l) c -> (b f) l c", f=num_layers).contiguous()
        return res


class CrossAttnProcessor:
    """Standard processor for the cross-attention (``attn2``) layers, ignoring the LSA arguments."""

    def __init__(self, chained_proc):
        self.chained_proc = chained_proc

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, num_layers=1):
        return self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)


class SPUNet(nn.Module):
    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        self.unet = unet

        procs = {}
        for name in unet.attn_processors.keys():
            default_proc = XFormersAttnProcessor() if is_xformers_available() else AttnProcessor2_0()
            wrapper = LayerwiseAttnProcessor if name.endswith("attn1.processor") else CrossAttnProcessor
            procs[name] = wrapper(default_proc)
        unet.set_attn_processor(procs)

        self.positional_encoder = PositionalEncoding(unet.config.cross_attention_dim, max_len=10)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward(self, sample, timestep, encoder_hidden_states, num_layers, added_cond_kwargs=None):
        """
        Args:
            sample: noisy SP map latents, (B * num_layers, 4, h, w).
            timestep: (B,) during training or a scalar tensor during inference.
            encoder_hidden_states: DINOv2 tokens of the condition image, (B, l, c).
            num_layers: number of SP layers per object.
            added_cond_kwargs: SDXL added conditions (``text_embeds`` and ``time_ids``), batch size B.
        """
        if added_cond_kwargs is None:
            added_cond_kwargs = {}
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=sample.device)
        timestep = timestep.long()
        if timestep.ndim > 0:
            timestep = repeat(timestep, "b -> (b f)", f=num_layers)

        # SDXL added conditions, the size conditioning is set to the SP map resolution
        target_added_cond = {}
        if "text_embeds" in added_cond_kwargs:
            target_added_cond["text_embeds"] = repeat(added_cond_kwargs["text_embeds"], "b c -> (b f) c",
                                                      f=num_layers).contiguous()
        if "time_ids" in added_cond_kwargs:
            time_ids = added_cond_kwargs["time_ids"]
            res = sample.shape[-2] * 8
            time_id = torch.tensor([res, res, 0, 0, res, res], device=time_ids.device, dtype=time_ids.dtype)
            target_added_cond["time_ids"] = repeat(time_id, "c -> (b f) c", b=time_ids.shape[0],
                                                   f=num_layers).contiguous()

        # every SP layer is conditioned on the image tokens plus a layer encoding
        image_embeds = repeat(encoder_hidden_states, "b l c -> b f l c", f=num_layers)
        image_embeds = rearrange(self.positional_encoder(image_embeds, offset=1), "b f l c -> (b f) l c")

        return self.unet(
            sample, timestep, image_embeds,
            class_labels=None,
            cross_attention_kwargs=dict(num_layers=num_layers),
            added_cond_kwargs=target_added_cond,
        )
