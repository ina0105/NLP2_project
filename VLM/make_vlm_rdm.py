import os
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

# --- Output folder ---
OUTPUT_DIR = "rdm_output_blip2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Context configurations ---
CONFIGS = {
    "vlm_five_general_representations": {
        "rep_dir": "/home/scur2169/NLP2_project/vlm_five_general_representations",
        "rdm_path": os.path.join(OUTPUT_DIR, "vlm_five_general_rdm.npy"),
    },
    "vlm_five_distinct_representations": {
        "rep_dir": "/home/scur2169/NLP2_project/vlm_five_distinct_representations",
        "rdm_path": os.path.join(OUTPUT_DIR, "vlm_five_distinct_rdm.npy"),
    },
    "vlm_one_word_representations": {
        "rep_dir": "/home/scur2169/NLP2_project/vlm_one_word_representations",
        "rdm_path": os.path.join(OUTPUT_DIR, "vlm_one_word_rdm.npy"),
    },
    "vlm_one_distinct_representations": {
        "rep_dir": "/home/scur2169/NLP2_project/vlm_one_distinct_representations",
        "rdm_path": os.path.join(OUTPUT_DIR, "vlm_one_distinct_rdm.npy"),
    },
    "vlm_one_general_representations": {
        "rep_dir": "/home/scur2169/NLP2_project/vlm_one_general_representations",
        "rdm_path": os.path.join(OUTPUT_DIR, "vlm_one_general_rdm.npy"),
    }
}


def compute_rdm(rep_dir, rdm_path):
    # --- Collect words from filenames ---
    words = [f.replace(".pt", "") for f in os.listdir(rep_dir) if f.endswith(".pt")]
    words.sort()

    # --- Load and average representations ---
    vectors = []
    for word in tqdm(words, desc=f"Processing: {os.path.basename(rep_dir)}"):
        path = os.path.join(rep_dir, f"{word}.pt")
        rep_dict = torch.load(path)

        if not rep_dict:
            raise ValueError(f"No representations found for word '{word}' in file: {path}")

        all_vectors = torch.stack(list(rep_dict.values()))  # [n_templates, hidden_dim]
        avg_vector = all_vectors.mean(dim=0)
        vectors.append(avg_vector.cpu().to(torch.float32).numpy())

    vectors = np.stack(vectors)  # [n_words, hidden_dim]

    # --- Compute cosine RDM ---
    distances = pdist(vectors, metric="cosine")
    rdm = squareform(distances)  # [n_words, n_words]

    # --- Save RDM ---
    np.save(rdm_path, rdm)
    print(f"Saved RDM to: {rdm_path}")


if __name__ == "__main__":
    for config_name, cfg in CONFIGS.items():
        print(f"\n--- Generating RDM for: {config_name} ---")
        compute_rdm(cfg["rep_dir"], cfg["rdm_path"])
