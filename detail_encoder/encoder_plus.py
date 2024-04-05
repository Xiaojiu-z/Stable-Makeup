from typing import List
import torch
from torchvision import transforms
from transformers import CLIPImageProcessor
from transformers import CLIPVisionModel as OriginalCLIPVisionModel
from ._clip import CLIPVisionModel
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import os

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")
if is_torch2_available():
    from .attention_processor import SSRAttnProcessor2_0 as SSRAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from .attention_processor import SSRAttnProcessor, AttnProcessor
from .resampler import Resampler

class detail_encoder(torch.nn.Module):
    """from SSR-encoder"""
    def __init__(self, unet, image_encoder_path, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # load image encoder
        clip_encoder = OriginalCLIPVisionModel.from_pretrained(image_encoder_path)
        self.image_encoder = CLIPVisionModel(clip_encoder.config)
        state_dict = clip_encoder.state_dict()
        self.image_encoder.load_state_dict(state_dict, strict=False)
        self.image_encoder.to(self.device, self.dtype)
        del clip_encoder
        self.clip_image_processor = CLIPImageProcessor()

        # load SSR layers
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = SSRAttnProcessor(hidden_size=hidden_size, cross_attention_dim=1024, scale=1).to(self.device, dtype=self.dtype)
        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        self.SSR_layers = adapter_modules
        self.SSR_layers.to(self.device, dtype=self.dtype)
        self.resampler = self.init_proj()

    def init_proj(self):
        resampler = Resampler().to(self.device, dtype=self.dtype)
        return resampler

    def forward(self, img):
        image_embeds = self.image_encoder(img, output_hidden_states=True)['hidden_states'][2::2]
        image_embeds = torch.cat(image_embeds, dim=1)
        image_embeds = self.resampler(image_embeds)
        return image_embeds

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = []
        for pil in pil_image:
            tensor_image = self.clip_image_processor(images=pil, return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)
            clip_image.append(tensor_image)
        clip_image = torch.cat(clip_image, dim=0)

        # cond
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True)['hidden_states'][2::2]  # 1 257*12 1024
        clip_image_embeds = torch.cat(clip_image_embeds, dim=1)
        uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True)['hidden_states'][2::2]
        uncond_clip_image_embeds = torch.cat(uncond_clip_image_embeds, dim=1)
        clip_image_embeds = self.resampler(clip_image_embeds)
        uncond_clip_image_embeds = self.resampler(uncond_clip_image_embeds)
        return clip_image_embeds, uncond_clip_image_embeds

    def generate(
            self,
            id_image,
            makeup_image,
            seed=None,
            guidance_scale=2,
            num_inference_steps=30,
            pipe=None,
            **kwargs,
    ):
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(makeup_image)

        prompt_embeds = image_prompt_embeds
        negative_prompt_embeds = uncond_image_prompt_embeds

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        image = pipe(
            image=id_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images[0]

        return image