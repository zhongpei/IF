from __future__ import annotations

import gc
import json
import tempfile
from typing import Generator

import numpy as np
import PIL.Image
import torch
from diffusers import DiffusionPipeline, StableDiffusionUpscalePipeline
from diffusers.pipelines.deepfloyd_if import (fast27_timesteps,
                                              smart27_timesteps,
                                              smart50_timesteps,
                                              smart100_timesteps,
                                              smart185_timesteps)

from settings import (DISABLE_AUTOMATIC_CPU_OFFLOAD, DISABLE_SD_X4_UPSCALER,
                      HF_TOKEN, MAX_NUM_IMAGES, MAX_NUM_STEPS, MAX_SEED,
                      RUN_GARBAGE_COLLECTION)


class Model:
    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.pipe = None
        self.super_res_1_pipe = None
        self.super_res_2_pipe = None
        self.watermark_image = None

        if torch.cuda.is_available():
            self.load_weights()
            self.watermark_image = PIL.Image.fromarray(
                self.pipe.watermarker.watermark_image.to(
                    torch.uint8).cpu().numpy(),
                mode='RGBA')

    def load_weights(self) -> None:
        self.pipe = DiffusionPipeline.from_pretrained(
            'DeepFloyd/IF-I-XL-v1.0',
            torch_dtype=torch.float16,
            variant='fp16',


            )
        self.super_res_1_pipe = DiffusionPipeline.from_pretrained(
            'DeepFloyd/IF-II-L-v1.0',
            text_encoder=None,
            torch_dtype=torch.float16,
            variant='fp16',

            )

        if not DISABLE_SD_X4_UPSCALER:
            self.super_res_2_pipe = StableDiffusionUpscalePipeline.from_pretrained(
                'stabilityai/stable-diffusion-x4-upscaler',
                torch_dtype=torch.float16
            )

        if DISABLE_AUTOMATIC_CPU_OFFLOAD:
            self.pipe.to(self.device)
            self.super_res_1_pipe.to(self.device)
            if not DISABLE_SD_X4_UPSCALER:
                self.super_res_2_pipe.to(self.device)
        else:
            self.pipe.enable_model_cpu_offload()
            self.super_res_1_pipe.enable_model_cpu_offload()
            if not DISABLE_SD_X4_UPSCALER:
                self.super_res_2_pipe.enable_model_cpu_offload()

    def apply_watermark_to_sd_x4_upscaler_results(
            self, images: list[PIL.Image.Image]) -> None:
        w, h = images[0].size

        stability_x4_upscaler_sample_size = 128

        coef = min(h / stability_x4_upscaler_sample_size,
                   w / stability_x4_upscaler_sample_size)
        img_h, img_w = (int(h / coef), int(w / coef)) if coef < 1 else (h, w)

        S1, S2 = 1024**2, img_w * img_h
        K = (S2 / S1)**0.5
        watermark_size = int(K * 62)
        watermark_x = img_w - int(14 * K)
        watermark_y = img_h - int(14 * K)

        watermark_image = self.watermark_image.copy().resize(
            (watermark_size, watermark_size),
            PIL.Image.Resampling.BICUBIC,
            reducing_gap=None)

        for image in images:
            image.paste(watermark_image,
                        box=(
                            watermark_x - watermark_size,
                            watermark_y - watermark_size,
                            watermark_x,
                            watermark_y,
                        ),
                        mask=watermark_image.split()[-1])

    @staticmethod
    def to_pil_images(images: torch.Tensor) -> list[PIL.Image.Image]:
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = np.round(images * 255).astype(np.uint8)
        return [PIL.Image.fromarray(image) for image in images]

    @staticmethod
    def check_seed(seed: int) -> None:
        if not 0 <= seed <= MAX_SEED:
            raise ValueError

    @staticmethod
    def check_num_images(num_images: int) -> None:
        if not 1 <= num_images <= MAX_NUM_IMAGES:
            raise ValueError

    @staticmethod
    def check_num_inference_steps(num_steps: int) -> None:
        if not 1 <= num_steps <= MAX_NUM_STEPS:
            raise ValueError

    @staticmethod
    def get_custom_timesteps(name: str) -> list[int] | None:
        if name == 'none':
            timesteps = None
        elif name == 'fast27':
            timesteps = fast27_timesteps
        elif name == 'smart27':
            timesteps = smart27_timesteps
        elif name == 'smart50':
            timesteps = smart50_timesteps
        elif name == 'smart100':
            timesteps = smart100_timesteps
        elif name == 'smart185':
            timesteps = smart185_timesteps
        else:
            raise ValueError
        return timesteps

    @staticmethod
    def run_garbage_collection():
        gc.collect()
        torch.cuda.empty_cache()

    def run_stage1(
        self,
        prompt: str,
        negative_prompt: str = '',
        seed: int = 0,
        num_images: int = 1,
        guidance_scale_1: float = 7.0,
        custom_timesteps_1: str = 'smart100',
        num_inference_steps_1: int = 100,
    ) -> tuple[list[PIL.Image.Image], str, str]:
        self.check_seed(seed)
        self.check_num_images(num_images)
        self.check_num_inference_steps(num_inference_steps_1)

        if RUN_GARBAGE_COLLECTION:
            self.run_garbage_collection()

        generator = torch.Generator(device=self.device).manual_seed(seed)

        prompt_embeds, negative_embeds = self.pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt)

        timesteps = self.get_custom_timesteps(custom_timesteps_1)

        images = self.pipe(prompt_embeds=prompt_embeds,
                           negative_prompt_embeds=negative_embeds,
                           num_images_per_prompt=num_images,
                           guidance_scale=guidance_scale_1,
                           timesteps=timesteps,
                           num_inference_steps=num_inference_steps_1,
                           generator=generator,
                           output_type='pt').images
        pil_images = self.to_pil_images(images)
        self.pipe.watermarker.apply_watermark(
            pil_images, self.pipe.unet.config.sample_size)

        stage1_params = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'seed': seed,
            'num_images': num_images,
            'guidance_scale_1': guidance_scale_1,
            'custom_timesteps_1': custom_timesteps_1,
            'num_inference_steps_1': num_inference_steps_1,
        }
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as param_file:
            param_file.write(json.dumps(stage1_params))
        stage1_result = {
            'prompt_embeds': prompt_embeds,
            'negative_embeds': negative_embeds,
            'images': images,
            'pil_images': pil_images,
        }
        with tempfile.NamedTemporaryFile(delete=False) as result_file:
            torch.save(stage1_result, result_file.name)
        return pil_images, param_file.name, result_file.name

    def run_stage2(
        self,
        stage1_result_path: str,
        stage2_index: int,
        seed_2: int = 0,
        guidance_scale_2: float = 4.0,
        custom_timesteps_2: str = 'smart50',
        num_inference_steps_2: int = 50,
        disable_watermark: bool = False,
    ) -> PIL.Image.Image:
        self.check_seed(seed_2)
        self.check_num_inference_steps(num_inference_steps_2)

        if RUN_GARBAGE_COLLECTION:
            self.run_garbage_collection()

        generator = torch.Generator(device=self.device).manual_seed(seed_2)

        stage1_result = torch.load(stage1_result_path)
        prompt_embeds = stage1_result['prompt_embeds']
        negative_embeds = stage1_result['negative_embeds']
        images = stage1_result['images']
        images = images[[stage2_index]]

        timesteps = self.get_custom_timesteps(custom_timesteps_2)

        out = self.super_res_1_pipe(image=images,
                                    prompt_embeds=prompt_embeds,
                                    negative_prompt_embeds=negative_embeds,
                                    num_images_per_prompt=1,
                                    guidance_scale=guidance_scale_2,
                                    timesteps=timesteps,
                                    num_inference_steps=num_inference_steps_2,
                                    generator=generator,
                                    output_type='pt',
                                    noise_level=250).images
        pil_images = self.to_pil_images(out)

        if disable_watermark:
            return pil_images[0]

        self.super_res_1_pipe.watermarker.apply_watermark(
            pil_images, self.super_res_1_pipe.unet.config.sample_size)
        return pil_images[0]

    def run_stage3(
        self,
        image: PIL.Image.Image,
        prompt: str = '',
        negative_prompt: str = '',
        seed_3: int = 0,
        guidance_scale_3: float = 9.0,
        num_inference_steps_3: int = 75,
    ) -> PIL.Image.Image:
        self.check_seed(seed_3)
        self.check_num_inference_steps(num_inference_steps_3)

        if RUN_GARBAGE_COLLECTION:
            self.run_garbage_collection()

        generator = torch.Generator(device=self.device).manual_seed(seed_3)
        out = self.super_res_2_pipe(image=image,
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    num_images_per_prompt=1,
                                    guidance_scale=guidance_scale_3,
                                    num_inference_steps=num_inference_steps_3,
                                    generator=generator,
                                    noise_level=100).images
        self.apply_watermark_to_sd_x4_upscaler_results(out)
        return out[0]

    def run_stage2_3(
        self,
        stage1_result_path: str,
        stage2_index: int,
        seed_2: int = 0,
        guidance_scale_2: float = 4.0,
        custom_timesteps_2: str = 'smart50',
        num_inference_steps_2: int = 50,
        prompt: str = '',
        negative_prompt: str = '',
        seed_3: int = 0,
        guidance_scale_3: float = 9.0,
        num_inference_steps_3: int = 75,
    ) -> Generator[PIL.Image.Image]:
        self.check_seed(seed_3)
        self.check_num_inference_steps(num_inference_steps_3)

        out_image = self.run_stage2(
            stage1_result_path=stage1_result_path,
            stage2_index=stage2_index,
            seed_2=seed_2,
            guidance_scale_2=guidance_scale_2,
            custom_timesteps_2=custom_timesteps_2,
            num_inference_steps_2=num_inference_steps_2,
            disable_watermark=True)
        temp_image = out_image.copy()
        self.super_res_1_pipe.watermarker.apply_watermark(
            [temp_image], self.super_res_1_pipe.unet.config.sample_size)
        yield temp_image
        yield self.run_stage3(image=out_image,
                              prompt=prompt,
                              negative_prompt=negative_prompt,
                              seed_3=seed_3,
                              guidance_scale_3=guidance_scale_3,
                              num_inference_steps_3=num_inference_steps_3)