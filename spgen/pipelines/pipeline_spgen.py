"""Image-conditioned generation of multi-layer SP maps."""
import torch
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from PIL import Image

from spgen.models.sp_unet import SPUNet


def load_schedulers(path):
    """Training (DDIM, zero terminal SNR) and sampling (Euler Ancestral) schedulers, both with v-prediction."""
    train_scheduler = DDIMScheduler.from_pretrained(path, subfolder="scheduler", rescale_betas_zero_snr=True,
                                                    prediction_type="v_prediction")
    sample_scheduler = EulerAncestralDiscreteScheduler.from_config(
        train_scheduler.config, timestep_spacing="trailing", prediction_type="v_prediction",
        rescale_betas_zero_snr=False)
    return train_scheduler, sample_scheduler


def encode_image_tokens(image_encoder, pixel_values):
    """DINOv2 tokens duplicated along the channel dim to match the SDXL cross-attention dim (2 x 1024)."""
    tokens = image_encoder(pixel_values, output_hidden_states=False).last_hidden_state
    return torch.cat([tokens, tokens], dim=-1)


def pooled_embed_dim(unet_config):
    """Dimension of the SDXL pooled text embedding (unused, fed with zeros)."""
    return unet_config.projection_class_embeddings_input_dim - 6 * unet_config.addition_time_embed_dim


class SPGenPipeline:
    def __init__(self, vae, unet: SPUNet, scheduler, image_encoder, image_processor):
        """
        Args:
            vae: finetuned single-channel SP map AutoEncoder.
            unet: ``SPUNet`` wrapping the finetuned SDXL UNet.
            scheduler: sampling scheduler (Euler Ancestral in the paper).
            image_encoder: DINOv2 model.
            image_processor: DINOv2 image processor.
        """
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.image_encoder = image_encoder
        self.image_processor = image_processor

    @property
    def device(self):
        return self.unet.unet.device

    @torch.no_grad()
    def __call__(self, image: Image.Image, num_layers=4, height=256, width=512, num_inference_steps=40,
                 guidance_scale=4.5, generator=None):
        """
        Returns:
            sp_maps: (num_layers, 1, height, width) tensor in [-1, 1].
            latents: (num_layers, 4, height / 8, width / 8) tensor.
        """
        device = self.device
        unet_dtype = self.unet.unet.dtype
        do_cfg = guidance_scale > 1.0

        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device, self.image_encoder.dtype)
        image_embeds = encode_image_tokens(self.image_encoder, pixel_values).to(unet_dtype)
        if do_cfg:
            image_embeds = torch.cat([torch.zeros_like(image_embeds), image_embeds])
        batch = image_embeds.shape[0]

        pooled_dim = pooled_embed_dim(self.unet.unet.config)
        added_cond_kwargs = {
            "text_embeds": torch.zeros(batch, pooled_dim, device=device, dtype=unet_dtype),
            "time_ids": torch.tensor([[height, width, 0, 0, height, width]] * batch, device=device, dtype=unet_dtype),
        }

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        shape = (num_layers, self.unet.unet.config.in_channels, height // 8, width // 8)
        latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        latents = latents * self.scheduler.init_noise_sigma

        for t in self.scheduler.timesteps:
            latent_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_input = self.scheduler.scale_model_input(latent_input, t).to(unet_dtype)
            noise_pred = self.unet(latent_input, t, image_embeds, num_layers=num_layers,
                                   added_cond_kwargs=added_cond_kwargs).sample.float()
            if do_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

        latents = latents.to(self.vae.dtype)
        sp_maps = torch.cat([self.vae.decode(lat[None] / self.vae.config.scaling_factor).sample for lat in latents])
        return sp_maps.clamp(-1, 1), latents
