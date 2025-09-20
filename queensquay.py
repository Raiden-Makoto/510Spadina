import clip
import torch
from PIL import Image
import os
import faiss
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

characters = "./GenshinCharacters"
image_files = [os.path.join(characters, f) for f in os.listdir(characters) if f.endswith((".png", ".jpg", ".jpeg"))]

image_embeddings = []
for F in image_files:
    image = preprocess(Image.open(F)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    image_embeddings.append(embedding.cpu().numpy())
image_embeddings = np.vstack(image_embeddings).astype(np.float32)

index = faiss.IndexFlat(image_embeddings.shape[1])
index.add(image_embeddings)

query = input("Enter a query: ").strip()
text = clip.tokenize([query]).to(device)

with torch.no_grad():
    text_embedding = model.encode_text(text)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding.cpu().numpy()

D, I = index.search(text_embedding, k=5)

print("Closest Matches:")
for i, idx in enumerate(I[0]):
    print(f"{i+1}. {image_files[idx]} (score: {D[0][i]:.4f})")
