import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import pdb
import itertools
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPImageProcessor

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from pipeline_sd15 import StableDiffusionControlNetPipeline
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from utils.unet_with_adapter import UNet2DConditionModel as UNet2DConditionModelWithAdapter
from detail_encoder.encoder_plus import detail_encoder
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.23.0")

logger = get_logger(__name__)

def concatenate_images(image_files, output_file):
    images = image_files  # list
    max_height = max(img.height for img in images)
    images = [img.resize((img.width, max_height)) for img in images]
    total_width = sum(img.width for img in images)
    combined = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    combined.save(output_file)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def log_validation(vae, text_encoder, tokenizer, unet, controlnet_id, controlnet_pose, makeup_encoder, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")
    controlnet_id = accelerator.unwrap_model(controlnet_id)
    controlnet_pose = accelerator.unwrap_model(controlnet_pose)
    makeup_encoder = accelerator.unwrap_model(makeup_encoder)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=[controlnet_id, controlnet_pose],
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    validation_ids = args.validation_ids
    validation_makeups = args.validation_makeups
    validation_poses = args.validation_poses
    image_logs = []
    validation_path = os.path.join(args.output_dir, "validation", f"step-{step}")
    os.makedirs(validation_path, exist_ok=True)
    _num = 0
    for validation_id, validation_makeup, validation_pose in zip(validation_ids, validation_makeups, validation_poses):
        _num += 1
        validation_id = Image.open(validation_id).convert("RGB").resize((512, 512))
        validation_makeup = Image.open(validation_makeup).convert("RGB").resize((512, 512))
        validation_pose = Image.open(validation_pose).convert("RGB").resize((512, 512))
        images = []
        for num in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = makeup_encoder.generate(id_image=[validation_id, validation_pose], makeup_image=validation_makeup, pipe=pipeline)
                concatenate_images([validation_pose, validation_id, validation_makeup, image], os.path.join(validation_path, str(num)+str(_num)+".jpg"))
            images.append(image)
        image_logs.append(
            {"validation_id": validation_id, "validation_makeup": validation_makeup, "validation_results": images}
        )


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="sd_model_v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument("--makeup_column", type=str, default="aug_mu"),
    parser.add_argument("--warp_column", type=str, default="makeup_image"),
    parser.add_argument("--id_column", type=str, default="aug_id"),
    parser.add_argument("--pose_column", type=str, default="pose"),

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="train_lr5e-5",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="/share2/zhangyuxuan/project/train_ip_cn/datasets/makeup_data.jsonl",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_ids",
        type=str,
        default=["/share2/zhangyuxuan/project/train_ip_cn/datasets/debug_dataset/origin/orig_2.png"],
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_makeups",
        type=str,
        default=["/share2/zhangyuxuan/project/train_ip_cn/datasets/debug_dataset/warp/warp_2.png"],
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=3,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_poses",
        type=str,
        default=["/share2/zhangyuxuan/project/train_ip_cn/datasets/debug_dataset/warp/warp_2.png"],
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=10,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def make_train_dataset(args, tokenizer, accelerator):

    if args.train_data_dir is not None:
        dataset = load_dataset('json', data_files=args.train_data_dir)
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.makeup_column is None:
        makeup_column = column_names[0]
        logger.info(f"image column defaulting to {makeup_column}")
    else:
        makeup_column = args.makeup_column
        if makeup_column not in column_names:
            raise ValueError(
                f"`--makeup_image_column` value '{args.makeup_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
    if args.id_column is None:
        id_column = column_names[1]
        logger.info(f"id column defaulting to {id_column}")
    else:
        id_column = args.id_column
        if id_column not in column_names:
            raise ValueError(
                f"`--id_column` value '{args.id_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
    if args.warp_column is None:
        warp_column = column_names[1]
        logger.info(f" column defaulting to {warp_column}")
    else:
        warp_column = args.warp_column
        if warp_column not in column_names:
            raise ValueError(
                f"`--warp_column` value '{args.warp_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.pose_column is None:
        pose_column = column_names[1]
        logger.info(f" column defaulting to {pose_column}")
    else:
        pose_column = args.pose_column
        if pose_column not in column_names:
            raise ValueError(
                f"`--pose_column` value '{args.pose_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    prob = 0.7
    OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    height = 512
    width = 512
    channels = 3
    black_image = np.zeros((height, width, channels), dtype=np.uint8)

    clip_norm = transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
    norm = transforms.Normalize([0.5], [0.5])
    to_tensor = transforms.ToTensor()

    pixel_transform = A.Compose([
        A.SmallestMaxSize(max_size=512),
        A.CenterCrop(512, 512),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=None, translate_percent=(-0.1, 0.1), rotate=(-10, 10), p=0.5),
    ], additional_targets={'image0': 'image', 'image1': 'image', 'image2': 'image'})

    clip_transform = A.Compose([
        A.SmallestMaxSize(max_size=224),
        A.CenterCrop(224, 224),
        A.Affine(scale=(0.5, 1.2), translate_percent=(-0.1, 0.1), rotate=(-10, 10), p=0.7),
        A.OneOf(
            [
                A.PixelDropout(dropout_prob=0.1, p=prob),
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=prob),
                A.GaussianBlur(blur_limit=(3, 5), p=prob),
            ]
        )]
    )

    def clip_imgaug(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = clip_transform(image=image)
        image = clip_norm(to_tensor(results["image"]/255.))
        return image

    def imgaug(id_image, mu_image, pose):
        id_image = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)
        mu_image = cv2.cvtColor(mu_image, cv2.COLOR_BGR2RGB)
        pose = cv2.cvtColor(pose, cv2.COLOR_BGR2RGB)
        if random.random() < 0.05:
            pose = black_image
        results = pixel_transform(image=id_image, image0=mu_image, image1=pose)
        id_image, mu_image, pose = to_tensor(results["image"]/255.), norm(to_tensor(results["image0"]/255.)), to_tensor(results["image1"]/255.)
        return id_image, mu_image, pose

    def preprocess_train(examples):

        id_images = [cv2.imread(image) for image in examples[id_column]]
        makeup_images = [cv2.imread(image) for image in examples[makeup_column]]
        pose_images = [cv2.imread(image) for image in examples[pose_column]]

        pair = [imgaug(image1, image2, image3) for image1, image2, image3 in zip(id_images, makeup_images, pose_images)]
        id_images, makeup_images, pose_images = zip(*pair)
        id_images_ls = list(id_images)
        makeup_images_ls = list(makeup_images)
        pose_images_ls = list(pose_images)

        warp_images = [cv2.imread(image) for image in examples[warp_column]]
        warp_images = [clip_imgaug(image) for image in warp_images]

        examples["id_pixel_values"] = id_images_ls
        examples["warp_pixel_values"] = warp_images
        examples["makeup_pixel_values"] = makeup_images_ls
        examples["pose_pixel_values"] = pose_images_ls

        return examples

    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    warp_pixel_values = torch.stack([example["warp_pixel_values"] for example in examples])
    warp_pixel_values = warp_pixel_values.to(memory_format=torch.contiguous_format).float()
    makeup_pixel_values = torch.stack([example["makeup_pixel_values"] for example in examples])
    makeup_pixel_values = makeup_pixel_values.to(memory_format=torch.contiguous_format).float()
    id_pixel_values = torch.stack([example["id_pixel_values"] for example in examples])
    id_pixel_values = id_pixel_values.to(memory_format=torch.contiguous_format).float()
    pose_pixel_values = torch.stack([example["pose_pixel_values"] for example in examples])
    pose_pixel_values = pose_pixel_values.to(memory_format=torch.contiguous_format).float()
    return {
        "warp_pixel_values": warp_pixel_values,
        "id_pixel_values": id_pixel_values,
        "makeup_pixel_values": makeup_pixel_values,
        "pose_pixel_values": pose_pixel_values,
    }

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = OriginalUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet_id = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        controlnet_pose = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet_id = ControlNetModel.from_unet(unet)
        controlnet_pose = ControlNetModel.from_unet(unet)

    image_encoder_path = "./models/image_encoder_l"
    makeup_encoder = detail_encoder(unet, image_encoder_path, accelerator.device, dtype=torch.float32)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet_id.requires_grad_(True)
    controlnet_pose.requires_grad_(True)
    makeup_encoder.SSR_layers.requires_grad_(True)
    makeup_encoder.resampler.requires_grad_(True)
    makeup_encoder.image_encoder.requires_grad_(False)

    optimizer_class = torch.optim.AdamW
    # Optimizer creation
    params_to_optimize = itertools.chain(controlnet_id.parameters(),
                                         controlnet_pose.parameters(),
                                         makeup_encoder.SSR_layers.parameters(),
                                         makeup_encoder.resampler.parameters())
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = make_train_dataset(args, tokenizer, accelerator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    makeup_encoder, controlnet_id, controlnet_pose, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        makeup_encoder, controlnet_id, controlnet_pose, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_makeups")
        tracker_config.pop("validation_ids")
        tracker_config.pop("validation_poses")
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    null_text_inputs = tokenizer(
        "", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids
    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(null_text_inputs.to(device=accelerator.device))[0]

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(makeup_encoder):

                # Convert images to latent space
                latents = vae.encode(batch["makeup_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                controlnet_image1 = batch["id_pixel_values"].to(dtype=weight_dtype)
                down_block_res_samples_id, mid_block_res_sample_id = controlnet_id(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.repeat(bsz, 1, 1),
                    controlnet_cond=controlnet_image1,
                    return_dict=False,
                )

                controlnet_image2 = batch["pose_pixel_values"].to(dtype=weight_dtype)
                down_block_res_samples_pose, mid_block_res_sample_pose = controlnet_pose(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.repeat(bsz, 1, 1),
                    controlnet_cond=controlnet_image2,
                    return_dict=False,
                )

                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples_id, down_block_res_samples_pose)
                ]
                mid_block_res_sample = mid_block_res_sample_id+mid_block_res_sample_pose

                image_embeds = makeup_encoder((batch["warp_pixel_values"]).to(accelerator.device, dtype=weight_dtype))
                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=image_embeds,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_ids is not None and global_step % args.validation_steps == 0:
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet_id,
                            controlnet_pose,
                            makeup_encoder,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
