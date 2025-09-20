import clip
import torch
from PIL import Image
import os
import faiss
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class FAISSMatcher:
    def __init__(self, device="cpu", characters_dir="./GenshinCharacters"):
        self.device = device
        self.characters_dir = characters_dir
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.image_files = [os.path.join(characters_dir, f) for f in os.listdir(characters_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        self.image_embeddings = []
        self.index = self.load_images()

    def load_images(self):
        for F in self.image_files:
            image = self.preprocess(Image.open(F)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_image(image)
                embedding /= embedding.norm(dim=-1, keepdim=True)
            self.image_embeddings.append(embedding.cpu().numpy())
        self.image_embeddings = np.vstack(self.image_embeddings).astype(np.float32)
        index = faiss.IndexFlat(self.image_embeddings.shape[1])
        index.add(self.image_embeddings)
        return index

    def query(self, query):
        text = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(text)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.cpu().numpy()
        D, I = self.index.search(text_embedding, k=5)
        print(f"Closest Match: {self.image_files[I[0][0]]} (score: {D[0][0]:.4f})")
        return self.image_files[I[0][0]]

if __name__ == "__main__":
    matcher = FAISSMatcher(device="cpu")
    query = input("Enter a query: ").strip()
    matcher.query(query)