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

    def find_character_by_name(self, name):
        """Find character file by name (case-insensitive partial match)"""
        name_lower = name.lower()
        
        # First try exact filename match
        for image_file in self.image_files:
            filename = os.path.basename(image_file)
            name_without_ext = os.path.splitext(filename)[0].lower()
            if name_lower == name_without_ext:
                return image_file
        
        # Then try partial matches
        matches = []
        for image_file in self.image_files:
            filename = os.path.basename(image_file)
            name_without_ext = os.path.splitext(filename)[0].lower()
            if name_lower in name_without_ext or name_without_ext in name_lower:
                matches.append(image_file)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"Multiple matches found for '{name}':")
            for i, match in enumerate(matches, 1):
                print(f"  {i}. {os.path.basename(match)}")
            while True:
                try:
                    choice = int(input("Select which one (1-{}): ".format(len(matches))))
                    if 1 <= choice <= len(matches):
                        return matches[choice - 1]
                except ValueError:
                    pass
                print("Invalid selection. Please try again.")
        
        return None

    def display_character_for_confirmation(self, image_path, character_name):
        """Display a single character image for confirmation"""
        plt.ion()
        plt.figure(figsize=(6, 6))
        img = Image.open(image_path).convert("RGB")
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Character: {character_name}\n{os.path.basename(image_path)}", fontsize=12)
        plt.tight_layout()
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except TypeError:
            plt.pause(0.1)

    def query(self, query, top_k=3):
        # Split query into individual attributes
        attributes = [attr.strip() for attr in query.split(',') if attr.strip()]
        
        if not attributes:
            print("No valid attributes provided.")
            return None
            
        print(f"Searching for attributes: {', '.join(attributes)}")
        
        # Start with all characters
        current_candidates = list(range(len(self.image_files)))
        
        # Recursively filter by each attribute
        for i, attribute in enumerate(attributes):
            print(f"Filtering by '{attribute}'...")
            
            # Get embeddings for current candidates
            if not current_candidates:
                print("No candidates remaining.")
                return None
                
            candidate_embeddings = self.embeddings[current_candidates]
            
            # Search for this attribute
            text = clip.tokenize([attribute]).to(self.device)
            with torch.no_grad():
                text_embedding = self.model.encode_text(text)
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
                text_embedding = text_embedding.cpu().numpy()
            
            # Create temporary index for current candidates
            temp_index = faiss.IndexFlatIP(candidate_embeddings.shape[1])
            temp_index.add(candidate_embeddings)
            
            # Find best matches for this attribute within current candidates
            D, I = temp_index.search(text_embedding, k=min(len(current_candidates), max(5, len(current_candidates)//2)))
            
            # Map back to original indices
            matching_indices = [current_candidates[idx] for idx in I[0]]
            
            current_candidates = matching_indices
            print(f"  Remaining candidates: {len(current_candidates)}")
        
        if not current_candidates:
            print("No characters match all attributes.")
            return None
        
        # Final ranking of remaining candidates
        print("Final ranking...")
        final_embeddings = self.embeddings[current_candidates]
        
        # Combine all attributes for final ranking
        combined_query = ", ".join(attributes)
        text = clip.tokenize([combined_query]).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(text)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.cpu().numpy()
        
        # Calculate final scores
        with torch.no_grad():
            candidate_vecs = torch.from_numpy(final_embeddings).to(self.device)
            text_vec = torch.from_numpy(text_embedding).to(self.device)
            text_vec = text_vec.to(candidate_vecs.dtype)
            sims = torch.matmul(candidate_vecs, text_vec.T).squeeze(1)
            sorted_indices = torch.argsort(sims, descending=True).tolist()
        
        # Map back to original indices and create final results
        final_candidates = [current_candidates[i] for i in sorted_indices]
        top_matches = [(self.image_files[idx], float(sims[i].item())) for i, idx in enumerate(final_candidates[:top_k])]
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
            raw = input(f"Select 1-{len(top_matches)} (or 0 to choose your own): ").strip()
            if raw.isdigit():
                num = int(raw)
                if 0 <= num <= len(top_matches):
                    choice = num
                    break
            print("Invalid selection. Please try again.")
        
        if choice == 0:
            # Close the triple preview window first
            try:
                plt.close('all')
            except Exception:
                pass
            
            # User wants to choose their own character
            character_name = input("Enter the character name: ").strip()
            selected_path = self.find_character_by_name(character_name)
            if selected_path:
                # Display the character image for confirmation
                self.display_character_for_confirmation(selected_path, character_name)
                confirm = input("Is this the character you want? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    print(f"Selected: {selected_path}")
                    return selected_path
                else:
                    print("Let's try again.")
                    return self.query(query, top_k)  # Recursive call to try again
            else:
                print(f"Character '{character_name}' not found. Please try again.")
                return self.query(query, top_k)  # Recursive call to try again
        else:
            selected_path = top_matches[choice - 1][0]
            # Close preview window after selection
            try:
                plt.close('all')
            except Exception:
                pass
            print(f"Selected: {selected_path}")
            return selected_path

if __name__ == "__main__":
    import sys
    
    matcher = FAISSMatcher(device="mps")
    
    # Check if query was provided as command line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Using query: {query}")
    else:
        query = input("Enter the desired qualities of your character, separated by commas. For example: cat ears, green eyes, sword ").strip()
    
    matcher.query(query, top_k=3)