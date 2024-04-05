import gradio as gr
import torch
import numpy as np
import requests
import random
from io import BytesIO
from utils import *
from constants import *
from inversion_utils import *
from modified_pipeline_semantic_stable_diffusion import SemanticStableDiffusionPipeline
from torch import autocast, inference_mode
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from transformers import AutoProcessor, BlipForConditionalGeneration
from share_btn import community_icon_html, loading_icon_html, share_js

from PIL import ImageFile
import random

# load pipelines
sd_model_id = "sd_model_v1-5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id).to(device)
sd_pipe.scheduler = DDIMScheduler.from_config(sd_model_id, subfolder="scheduler")

sega_pipe = SemanticStableDiffusionPipeline.from_pretrained(sd_model_id).to(device)

ImageFile.LOAD_TRUNCATED_IMAGES = True
input_image = "./examples/00001.png"  # @param

source_prompt = "human face"  # @param
target_prompt = "make up like a clown"  # @param
num_diffusion_steps = 100  # @param
source_guidance_scale = 0 # @param
reconstruct = True  # @param
skip_steps = 50  # @param
target_guidance_scale = 10 # @param

# SEGA only params
edit_concepts = ["star makeup", "heart makeup"]  # @param
edit_guidance_scales = [7, 15]  # @param
warmup_steps = [1, 1]  # @param
reverse_editing = [True, False]  # @param
thresholds = [0.95, 0.95]  # @param


def invert(x0: torch.FloatTensor, prompt_src: str = "", num_inference_steps=100, cfg_scale_src=3.5, eta=1):
    #  inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf,
    #  based on the code in https://github.com/inbarhub/DDPM_inversion

    #  returns wt, zs, wts:
    #  wt - inverted latent
    #  wts - intermediate inverted latents
    #  zs - noise maps

    sd_pipe.scheduler.set_timesteps(num_diffusion_steps)

    # vae encode image
    with autocast("cuda"), inference_mode():
        w0 = (sd_pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()

    # find Zs and wts - forward process
    wt, zs, wts = inversion_forward_process(sd_pipe, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src,
                                            prog_bar=True, num_inference_steps=num_diffusion_steps)
    return zs, wts


def sample(zs, wts, prompt_tar="", cfg_scale_tar=15, skip=36, eta=1):
    # reverse process (via Zs and wT)
    w0, _ = inversion_reverse_process(sd_pipe, xT=wts[skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar],
                                      prog_bar=True, zs=zs[skip:])

    # vae decode image
    with autocast("cuda"), inference_mode():
        x0_dec = sd_pipe.vae.decode(1 / 0.18215 * w0).sample
    if x0_dec.dim() < 4:
        x0_dec = x0_dec[None, :, :, :]
    img = image_grid(x0_dec)
    return img


def edit(wts, zs,
         tar_prompt="",
         steps=100,
         skip=36,
         tar_cfg_scale=8,
         edit_concept="",
         guidnace_scale=7,
         warmup=1,
         neg_guidance=False,
         threshold=0.95
         ):
    # SEGA
    # parse concepts and neg guidance
    editing_args = dict(
        editing_prompt=edit_concept,
        reverse_editing_direction=neg_guidance,
        edit_warmup_steps=warmup,
        edit_guidance_scale=guidnace_scale,
        edit_threshold=threshold,
        edit_momentum_scale=0.5,
        edit_mom_beta=0.6,
        eta=1,
    )
    latnets = wts[skip].expand(1, -1, -1, -1)
    sega_out = sega_pipe(prompt=tar_prompt, latents=latnets, guidance_scale=tar_cfg_scale,
                         num_images_per_prompt=1,
                         num_inference_steps=steps,
                         use_ddpm=True, wts=wts, zs=zs[skip:], **editing_args)
    return sega_out.images[0]

with open('prompt.txt') as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines]

for i in range(10000):
    try:
        input_image = 'origin_face' + str(i + 1) + '.png'
        randn = random.randint(0, len(lines)-1)
        target_prompt = lines[randn]
        x0 = load_512(input_image, device=device)
        # noise maps and latents
        zs, wts = invert(x0=x0, prompt_src=source_prompt, num_inference_steps=num_diffusion_steps,
                         cfg_scale_src=source_guidance_scale)
        if reconstruct:
            ddpm_out_img = sample(zs, wts, prompt_tar=target_prompt, skip=skip_steps, cfg_scale_tar=target_guidance_scale)
            ddpm_out_img.save(f'makeup/edit_{i+1}.png')
    except:
        continue






