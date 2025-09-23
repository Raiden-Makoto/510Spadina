import clip
import torch
from PIL import Image
import os
import faiss
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class FAISSMatcher:
    def __init__(self, device="mps", characters_dir="./GenshinCharacters"):
        self.device = device
        self.characters_dir = characters_dir
        # Use a larger CLIP backbone for better retrieval quality
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.image_files = [os.path.join(characters_dir, f) for f in os.listdir(characters_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        self.embeddings = []  # combined per-character embedding (image [+ optional text])
        self.index = self.load_images()

    def load_images(self):
        vectors = []
        for image_path in self.image_files:
            # Image embedding
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_emb = self.model.encode_image(image)
                image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

            # Optional text description embedding (same basename .txt)
            text_path = os.path.splitext(image_path)[0] + ".txt"
            if os.path.isfile(text_path):
                try:
                    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
                        description = f.read().strip()
                    if description:
                        tokens = clip.tokenize([description]).to(self.device)
                        with torch.no_grad():
                            text_emb = self.model.encode_text(tokens)
                            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                        # Fuse image and text embeddings (simple average), then renormalize
                        fused = (image_emb + text_emb) / 2.0
                        fused = fused / fused.norm(dim=-1, keepdim=True)
                        use_emb = fused
                    else:
                        use_emb = image_emb
                except Exception:
                    use_emb = image_emb
            else:
                use_emb = image_emb

            vectors.append(use_emb.cpu().numpy())

        self.embeddings = np.vstack(vectors).astype(np.float32)

        # Cosine similarity via inner-product on unit vectors
        index = faiss.IndexFlatIP(self.embeddings.shape[1])
        index.add(self.embeddings)
        return index

    def query(self, query, top_k=3):
        text = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(text)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.cpu().numpy()
        # Initial FAISS search (inner product ~= cosine on normalized vectors)
        initial_k = max(10, top_k)  # get more then rerank
        D, I = self.index.search(text_embedding, k=min(initial_k, len(self.image_files)))

        # Rerank with higher-precision cosine in torch
        with torch.no_grad():
            candidate_indices = I[0].tolist()
            candidate_vecs = torch.from_numpy(self.embeddings[candidate_indices]).to(self.device)
            text_vec = torch.from_numpy(text_embedding).to(self.device)  # shape (1, d)
            # Ensure matching dtypes for matmul (e.g., float32 vs float16)
            text_vec = text_vec.to(candidate_vecs.dtype)
            sims = torch.matmul(candidate_vecs, text_vec.T).squeeze(1)  # cosine since normalized
            sorted_indices = torch.argsort(sims, descending=True).tolist()

        reranked = [(self.image_files[candidate_indices[j]], float(sims[j].item())) for j in sorted_indices[:top_k]]
        top_matches = reranked
        # Display choices
        print("Top matches:")
        for idx, (path, score) in enumerate(top_matches, start=1):
            print(f"  {idx}. {path} (score: {score:.4f})")
        # Show previews in a single window (non-blocking)
        cols = len(top_matches)
        plt.ion()
        plt.figure(figsize=(4 * cols, 4))
        for idx, (path, score) in enumerate(top_matches, start=1):
            ax = plt.subplot(1, cols, idx)
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            ax.axis('off')
            title = f"{idx}. {os.path.basename(path)}\n{score:.4f}"
            ax.set_title(title, fontsize=10)
        plt.tight_layout()
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except TypeError:
            # Some backends ignore block kwarg; ensure a brief draw
            plt.pause(0.1)
        # Ask user to choose
        choice = None
        while choice is None:
            raw = input(f"Select 1-{len(top_matches)}: ").strip()
            if raw.isdigit():
                num = int(raw)
                if 1 <= num <= len(top_matches):
                    choice = num
                    break
            print("Invalid selection. Please try again.")
        selected_path = top_matches[choice - 1][0]
        # Close preview window after selection
        try:
            plt.close('all')
        except Exception:
            pass
        print(f"Selected: {selected_path}")
        return selected_path

if __name__ == "__main__":
    matcher = FAISSMatcher(device="mps")
    query = input("Enter the desired qualities of your character. For example, cat ears, green eyes, etc. ").strip()
    matcher.query(query, top_k=3)