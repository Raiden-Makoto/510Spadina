from diffusers import AutoencoderKL, StableDiffusionPipeline
import torch
from PIL import Image
import os
import sys
import gc
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model_id = "xyn-ai/anything-v4.0"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    safety_checker=None
).to(device)
#pipe.enable_xformers_memory_efficient_attention()
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float32
).to("mps")
pipe.vae = vae
pipe.enable_attention_slicing("max")  # uses the smallest possible slices (lowest VRAM, slowest)

def generate_image(file_path, optional_tags=None):
    selected_character = os.path.splitext(os.path.basename(file_path))[0].lower()
    # Handle empty optional tags
    if optional_tags:
        prompt = f"{selected_character}_(genshin impact), 1girl,{optional_tags}, portrait"
    else:
        prompt = f"{selected_character}_(genshin impact), 1girl, portrait"
    negative_prompt = "realistic, photorealistic, low quality, blur"
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        num_inference_steps=30,
        num_images_per_prompt=1,
    ).images[0]
    fname = f"Avatar_like_{selected_character}.png"
    result.save(fname)
    print(f"Image saved as {fname}")
    
    # Clear memory
    del result
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an avatar-like image using Anything V4 style transfer.")
    parser.add_argument("image_path", help="Path to the character image (e.g., ../GenshinCharacters/Fischl.png)")
    parser.add_argument("-t", "--optional-tags", dest="optional_tags", default=None,
                        help="Optional tags separated by commas (e.g., 'blonde, green eyes')")
    args = parser.parse_args()

    file_path = args.image_path
    generate_image(file_path, args.optional_tags)