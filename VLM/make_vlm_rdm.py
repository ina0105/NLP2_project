import os
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

# --- Config ---
REPRESENTATION_DIR = "/home/scur2169/NLP2_project/vlm_representations"
OUT_RDM_PATH = "vlm_rdm.npy"
OUT_WORD_LIST_PATH = "vlm_words.txt"

# --- Collect words from filenames ---
words = [f.replace(".pt", "") for f in os.listdir(REPRESENTATION_DIR) if f.endswith(".pt")]
words.sort()

# --- Load vectors and average across templates ---
vectors = []
for word in tqdm(words, desc="Loading and averaging representations"):
    path = os.path.join(REPRESENTATION_DIR, f"{word}.pt")
    rep_dict = torch.load(path)

    if not rep_dict:
        raise ValueError(f"No representations found for word '{word}' in file: {path}")

    all_vectors = torch.stack(list(rep_dict.values()))  # shape: [num_templates, hidden_dim]
    avg_vector = all_vectors.mean(dim=0)
    vectors.append(avg_vector.cpu().to(torch.float32).numpy())

vectors = np.stack(vectors)  # shape: [num_words, hidden_dim]

# --- Compute cosine RDM ---
distances = pdist(vectors, metric="cosine")
rdm = squareform(distances)  # shape: [num_words, num_words]

# --- Save outputs ---
np.save(OUT_RDM_PATH, rdm)
with open(OUT_WORD_LIST_PATH, "w") as f:
    for word in words:
        f.write(word + "\n")

print(f"✅ Saved RDM to: {OUT_RDM_PATH}")
print(f"✅ Saved word list to: {OUT_WORD_LIST_PATH}")
