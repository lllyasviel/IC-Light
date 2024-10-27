# Prediction interface for Cog ⚙️
# https://cog.run/python

import time
import os
import math
import torch
import tempfile
import mimetypes
import subprocess
import numpy as np
from typing import List
import safetensors.torch as sf
from cog import BasePredictor, Input, Path

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum

mimetypes.add_type("image/webp", ".webp")

MODEL_CACHE = "models"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs["cross_attention_kwargs"] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [
        [id_start] + tokens[i : i + chunk_length] + [id_end]
        for i in range(0, len(tokens), chunk_length)
    ]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = (
        torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    )  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process(
    input_fg,
    prompt,
    image_width,
    image_height,
    num_samples,
    seed,
    steps,
    a_prompt,
    n_prompt,
    cfg,
    highres_scale,
    highres_denoise,
    lowres_denoise,
    bg_source,
):
    bg_source = BGSource(bg_source)
    input_bg = None

    if bg_source == BGSource.NONE:
        pass
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise "Wrong initial latent!"

    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = (
        vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    )

    conds, unconds = encode_prompt_pair(
        positive_prompt=prompt + ", " + a_prompt, negative_prompt=n_prompt
    )

    if input_bg is None:
        latents = (
            t2i_pipe(
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=steps,
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type="latent",
                guidance_scale=cfg,
                cross_attention_kwargs={"concat_conds": concat_conds},
            ).images.to(vae.dtype)
            / vae.config.scaling_factor
        )
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = (
            i2i_pipe(
                image=bg_latent,
                strength=lowres_denoise,
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=int(round(steps / lowres_denoise)),
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type="latent",
                guidance_scale=cfg,
                cross_attention_kwargs={"concat_conds": concat_conds},
            ).images.to(vae.dtype)
            / vae.config.scaling_factor
        )

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [
        resize_without_crop(
            image=p,
            target_width=int(round(image_width * highres_scale / 64.0) * 64),
            target_height=int(round(image_height * highres_scale / 64.0) * 64),
        )
        for p in pixels
    ]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = (
        vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    )

    latents = (
        i2i_pipe(
            image=latents,
            strength=highres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / highres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type="latent",
            guidance_scale=cfg,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(vae.dtype)
        / vae.config.scaling_factor
    )

    pixels = vae.decode(latents).sample

    return pytorch2numpy(pixels)


@torch.inference_mode()
def process_relight(
    input_fg,
    prompt,
    image_width,
    image_height,
    num_samples,
    seed,
    steps,
    a_prompt,
    n_prompt,
    cfg,
    highres_scale,
    highres_denoise,
    lowres_denoise,
    bg_source,
):
    input_fg, matting = run_rmbg(input_fg)
    results = process(
        input_fg,
        prompt,
        image_width,
        image_height,
        num_samples,
        seed,
        steps,
        a_prompt,
        n_prompt,
        cfg,
        highres_scale,
        highres_denoise,
        lowres_denoise,
        bg_source,
    )
    return input_fg, results


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        global sd15_name, tokenizer, text_encoder, vae, unet, rmbg, new_conv_in, unet_original_forward, unet, model_path, sd_offset, sd_origin, keys, sd_merged, device, text_encoder, vae, unet, rmbg, unet, ddim_scheduler, euler_a_scheduler, dpmpp_2m_sde_karras_scheduler, t2i_pipe, i2i_pipe, quick_prompts, quick_subjects

        """Load the model into memory to make running multiple predictions efficient"""
        model_files = [
            "iclight_sd15_fc.safetensors",
            "models--briaai--RMBG-1.4.tar",
            "models--stablediffusionapi--realistic-vision-v51.tar",
        ]

        base_url = f"https://weights.replicate.delivery/default/IC-Light/{MODEL_CACHE}/"

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        for model_file in model_files:
            url = base_url + model_file

            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # 'stablediffusionapi/realistic-vision-v51'
        # 'runwayml/stable-diffusion-v1-5'
        sd15_name = "stablediffusionapi/realistic-vision-v51"
        tokenizer = CLIPTokenizer.from_pretrained(
            sd15_name,
            subfolder="tokenizer",
            cache_dir=MODEL_CACHE,
            # force_download=True,
            local_files_only=True,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            sd15_name,
            subfolder="text_encoder",
            cache_dir=MODEL_CACHE,
            # force_download=True,
            local_files_only=True,
        )
        vae = AutoencoderKL.from_pretrained(
            sd15_name,
            subfolder="vae",
            cache_dir=MODEL_CACHE,
            # force_download=True,
            local_files_only=True,
        )
        unet = UNet2DConditionModel.from_pretrained(
            sd15_name,
            subfolder="unet",
            cache_dir=MODEL_CACHE,
            # force_download=True,
            local_files_only=True,
        )
        rmbg = BriaRMBG.from_pretrained(
            "briaai/RMBG-1.4",
            cache_dir=MODEL_CACHE,
            # force_download=True,
            local_files_only=True,
        )

        # Change UNet

        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                8,
                unet.conv_in.out_channels,
                unet.conv_in.kernel_size,
                unet.conv_in.stride,
                unet.conv_in.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            new_conv_in.bias = unet.conv_in.bias
            unet.conv_in = new_conv_in

        unet_original_forward = unet.forward

        unet.forward = hooked_unet_forward

        # Load

        model_path = f"./{MODEL_CACHE}/iclight_sd15_fc.safetensors"

        sd_offset = sf.load_file(model_path)
        sd_origin = unet.state_dict()
        keys = sd_origin.keys()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged, keys

        # Device

        device = torch.device("cuda")
        text_encoder = text_encoder.to(device=device, dtype=torch.float16)
        vae = vae.to(device=device, dtype=torch.bfloat16)
        unet = unet.to(device=device, dtype=torch.float16)
        rmbg = rmbg.to(device=device, dtype=torch.float32)

        # SDP

        unet.set_attn_processor(AttnProcessor2_0())
        vae.set_attn_processor(AttnProcessor2_0())

        # Samplers

        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        euler_a_scheduler = EulerAncestralDiscreteScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, steps_offset=1
        )

        dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1,
        )

        # Pipelines

        t2i_pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=dpmpp_2m_sde_karras_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None,
        )

        i2i_pipe = StableDiffusionImg2ImgPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=dpmpp_2m_sde_karras_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None,
        )

        quick_prompts = [
            "sunshine from window",
            "neon light, city",
            "sunset over sea",
            "golden time",
            "sci-fi RGB glowing, cyberpunk",
            "natural lighting",
            "warm atmosphere, at home, bedroom",
            "magic lit",
            "evil, gothic, Yharnam",
            "light and shadow",
            "shadow from window",
            "soft studio lighting",
            "home atmosphere, cozy bedroom illumination",
            "neon, Wong Kar-wai, warm",
        ]
        quick_prompts = [[x] for x in quick_prompts]

        quick_subjects = [
            "beautiful woman, detailed face",
            "handsome man, detailed face",
        ]
        quick_subjects = [[x] for x in quick_subjects]

    def predict(
        self,
        subject_image: Path = Input(
            description="The main foreground image to be relighted"
        ),
        prompt: str = Input(
            description="A text description guiding the relighting and generation process"
        ),
        appended_prompt: str = Input(
            default="best quality",
            description="Additional text to be appended to the main prompt, enhancing image quality",
        ),
        negative_prompt: str = Input(
            default="lowres, bad anatomy, bad hands, cropped, worst quality",
            description="A text description of attributes to avoid in the generated images",
        ),
        width: int = Input(
            default=512,
            description="The width of the generated images in pixels",
            choices=[256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
        ),
        height: int = Input(
            default=640,
            description="The height of the generated images in pixels",
            choices=[256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
        ),
        steps: int = Input(
            default=25,
            description="The number of diffusion steps to perform during generation (more steps generally improves image quality but increases processing time)",
            ge=1,
            le=100,
        ),
        cfg: float = Input(
            default=2.0,
            description="Classifier-Free Guidance scale - higher values encourage adherence to prompt, lower values encourage more creative interpretation",
            ge=1.0,
            le=32.0,
        ),
        highres_scale: float = Input(
            default=1.5,
            description="The multiplier for the final output resolution relative to the initial latent resolution",
            ge=1.0,
            le=3.0,
        ),
        highres_denoise: float = Input(
            default=0.5,
            description="Controls the amount of denoising applied when refining the high resolution output (higher = more adherence to the upscaled latent, lower = more creative details added)",
            ge=0.1,
            le=1.0,
        ),
        lowres_denoise: float = Input(
            default=0.9,
            description="Controls the amount of denoising applied when generating the initial latent from the background image (higher = more adherence to the background, lower = more creative interpretation)",
            ge=0.1,
            le=1.0,
        ),
        light_source: str = Input(
            default=BGSource.NONE.value,
            description="The type and position of lighting to apply to the initial background latent",
            choices=[e.value for e in BGSource],
        ),
        seed: int = Input(
            description="A fixed random seed for reproducible results (omit this parameter for a randomized seed)",
            default=None,
        ),
        number_of_images: int = Input(
            default=1,
            description="The number of unique images to generate from the given input and settings",
            ge=1,
            le=12,
        ),
        output_format: str = Input(
            description="The image file format of the generated output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="The image compression quality (for lossy formats like JPEG and WebP). 100 = best quality, 0 = lowest quality.",
            default=80,
            ge=0,
            le=100,
        ),
    ) -> List[Path]:
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        input_fg = subject_image
        image_width = width
        image_height = height
        num_samples = number_of_images
        a_prompt = appended_prompt
        n_prompt = negative_prompt
        bg_source = light_source

        print(f"[!] ({type(input_fg)}) input_fg={input_fg}")
        print(f"[!] ({type(prompt)}) prompt={prompt}")
        print(f"[!] ({type(image_width)}) image_width={image_width}")
        print(f"[!] ({type(image_height)}) image_height={image_height}")
        print(f"[!] ({type(num_samples)}) num_samples={num_samples}")
        print(f"[!] ({type(seed)}) seed={seed}")
        print(f"[!] ({type(steps)}) steps={steps}")
        print(f"[!] ({type(a_prompt)}) a_prompt={a_prompt}")
        print(f"[!] ({type(n_prompt)}) n_prompt={n_prompt}")
        print(f"[!] ({type(cfg)}) cfg={cfg}")
        print(f"[!] ({type(highres_scale)}) highres_scale={highres_scale}")
        print(f"[!] ({type(highres_denoise)}) highres_denoise={highres_denoise}")
        print(f"[!] ({type(lowres_denoise)}) lowres_denoise={lowres_denoise}")
        print(f"[!] ({type(bg_source)}) bg_source={bg_source}")
        input_fg_np = np.array(Image.open(str(input_fg))) if input_fg else None

        with torch.inference_mode():
            output_bg, result_gallery = process_relight(
                input_fg_np,
                prompt,
                image_width,
                image_height,
                num_samples,
                seed,
                steps,
                a_prompt,
                n_prompt,
                cfg,
                highres_scale,
                highres_denoise,
                lowres_denoise,
                bg_source,
            )

        # Create a directory to save the output images
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)

        # Save the background image
        extension = output_format.lower()
        extension = "jpeg" if extension == "jpg" else extension
        # bg_path = os.path.join(output_dir, f"background.{extension}")
        save_params = {"format": extension.upper()}
        if output_format != "png":
            save_params["quality"] = output_quality
            save_params["optimize"] = True
        # Image.fromarray(output_bg).save(bg_path, **save_params)

        # Save the generated images
        output_paths = [] #[Path(bg_path)]
        for i, img in enumerate(result_gallery):
            img_path = os.path.join(output_dir, f"generated_{i}.{extension}")
            print(f"[~] Saving to {img_path}...")
            print(f"[~] Output format: {extension.upper()}")
            if output_format != "png":
                print(f"[~] Output quality: {output_quality}")

            Image.fromarray(img).save(img_path, **save_params)
            output_paths.append(Path(img_path))

        return output_paths
