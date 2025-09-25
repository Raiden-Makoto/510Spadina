#!/usr/bin/env python3
"""
A more accurate Human to Anime Feature Matcher
"""

import torch
import timm
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import clip
import glob
import sys
import argparse
from diffusers import AutoencoderKL, StableDiffusionPipeline
import gc

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Human to Anime Feature Matcher using DINO and CLIP')
    parser.add_argument('test_image', help='Path to the test image file')
    parser.add_argument('-t', '--optional-tags', dest='optional_tags', default=None,
                        help="Optional tags separated by commas (e.g., 'blonde, green eyes')")
    args = parser.parse_args()
    
    # Check if test image file exists
    if not os.path.exists(args.test_image):
        print(f"Error: Test image file '{args.test_image}' not found")
        sys.exit(1)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load models
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
    model = model.eval().to(device)
    clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    def get_dino_embedding(img_path):
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.forward_features(x)  # feature extraction
        return emb.cpu().numpy().flatten()

    def get_clip_embedding(img_path):
        img = Image.open(img_path).convert("RGB")
        img_pre = preprocess_clip(img).unsqueeze(0).to(device)
        with torch.no_grad():
            return clip_model.encode_image(img_pre).cpu().numpy().flatten()
    
    # Get all PNG and JPG files from GenshinCharacters directory
    avatar_files = glob.glob("./GenshinCharacters/*.png") + glob.glob("./GenshinCharacters/*.jpg")
    dino_embeddings = [get_dino_embedding(img) for img in avatar_files]
    clip_embeddings = [get_clip_embedding(img) for img in avatar_files]
    
    # Get test image path from command line argument
    test_path = args.test_image
    query_dino_emb = get_dino_embedding(test_path)
    query_clip_emb = get_clip_embedding(test_path)
    
    def combined_similarity(q_dino, q_clip, a_dino, a_clip, alpha=0.67):
        # normalize
        q_dino /= np.linalg.norm(q_dino)
        q_clip /= np.linalg.norm(q_clip)
        a_dino /= np.linalg.norm(a_dino)
        a_clip /= np.linalg.norm(a_clip)
        
        sim_dino = np.dot(q_dino, a_dino)
        sim_clip = np.dot(q_clip, a_clip)
        return alpha*sim_clip + (1-alpha)*sim_dino
    
    # Calculate similarities
    similarities = [combined_similarity(query_dino_emb, query_clip_emb, emb[0], emb[1]) for emb in zip(dino_embeddings, clip_embeddings)]
    
    # Find best match
    best_idx = int(np.argmax(similarities))
    # Print exact path only for downstream parsing compatibility
    print(avatar_files[best_idx])
    styletransfer_input = avatar_files[best_idx]

    sd_device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model_id = "xyn-ai/anything-v4.0"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(sd_device)
    #pipe.enable_xformers_memory_efficient_attention()
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(sd_device)
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
    
    generate_image(styletransfer_input, args.optional_tags)

if __name__ == "__main__":
    main()
