from diffusers import AutoencoderKL, StableDiffusionPipeline
import torch
from PIL import Image
import os
import sys
import gc

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

def generate_image(file_path):
    optional_tags = input("Enter optional tags, separated by commas, or press enter to skip.\nFor example, 'blonde, green eyes' ").strip()
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
    if len(sys.argv) < 2:
        print("Usage: python styletransfer.py <image_path>")
        print("Example: python styletransfer.py ../GenshinCharacters/Fischl.png")
        sys.exit(1)
    
    file_path = sys.argv[1]
    generate_image(file_path)