from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch
import os
from pathlib import Path
import gradio as gr

cache_dir = os.path.join(Path.cwd(), "cache")
# stage 1
stage_1 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    variant="fp16",
    torch_dtype=torch.float16,
    resume_download=True,

)
# stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
stage_1.enable_model_cpu_offload()

# stage 2
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0",
    text_encoder=None,
    variant="fp16",
    torch_dtype=torch.float16,
    resume_download=True,

)
# stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
stage_2.enable_model_cpu_offload()

# stage 3
safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker,
                  "watermarker": stage_1.watermarker}
stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    **safety_modules,
    torch_dtype=torch.float16,
    resume_download=True,

)
# stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
stage_3.enable_model_cpu_offload()


def gen(prompt, negative_prompt):
    # text embeds
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

    generator = torch.manual_seed(0)

    # stage 1
    image1 = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator,
                     output_type="pt").images
    # pt_to_pil(image1)[0].save("./if_stage_I.png")

    # stage 2
    image2 = stage_2(
        image=image1, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator,
        output_type="pt"
    ).images
    # pt_to_pil(image)[0].save("./if_stage_II.png")

    # stage 3
    image3 = stage_3(prompt=prompt, image=image2, generator=generator, noise_level=100, ).images
    # image3[0].save("./if_stage_III.png")
    return pt_to_pil(image1)[0], pt_to_pil(image2)[0], image3[0]


with gr.Blocks() as demo:
    prompt = gr.Text(label="prompt", lines=4, max_lines=6)
    negative_prompt = gr.Text(
        label='Negative prompt',
        show_label=False,
        lines=4,
        max_lines=6,
        value="high quality dslr photo, a photo product of a lemon inspired by natural and organic materials, wooden accents, intricately decorated with glowing vines of led lights, inspired by baroque luxury",
        placeholder='Enter a negative prompt',
        elem_id='negative-prompt-text-input',
    )
    # prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
    with gr.Row():
        gallery = gr.Gallery(
            label='results',
            show_label=False,
            elem_id='gallery'
        )

    gen_btn = gr.Button()
    gen_btn.click(
        gen,
        inputs=[prompt, negative_prompt],
        outputs=gallery
    )

demo.launch(show_api=False, share=False, enable_queue=False)
