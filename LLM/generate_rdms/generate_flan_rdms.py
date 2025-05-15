import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os

def load_flan_embeddings(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def compute_rdm(embeddings):
    # Convert embeddings to numpy array
    embedding_matrix = np.array(list(embeddings.values()))
    
    # Compute cosine distance RDM
    rdm = squareform(pdist(embedding_matrix, metric='cosine'))
    
    # Get upper triangle (excluding diagonal)
    rdm_up_tri = rdm[np.triu_indices(n=len(embeddings), k=1)]
    
    return rdm, rdm_up_tri

def save_rdms(rdm, rdm_up_tri, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as NumPy files
    np.save(f'{output_dir}/{prefix}_rdm.npy', rdm)
    np.save(f'{output_dir}/{prefix}_rdm_uptri.npy', rdm_up_tri)
    
    # Save as CSV files
    np.savetxt(f'{output_dir}/{prefix}_rdm.csv', rdm, delimiter=',')
    np.savetxt(f'{output_dir}/{prefix}_rdm_uptri.csv', rdm_up_tri, delimiter=',')

def main():
    # List of embedding JSON files to process
    embedding_files = [
        'output/flan_t5_one_word.json',
        'output/flan_t5_one_general.json',
        'output/flan_t5_one_distinct.json',
        'output/flan_t5_five_general.json',
        'output/flan_t5_five_distinct.json'
    ]
    output_dir = 'rdm_output/llm'
    for file_path in embedding_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        print(f"Processing {file_path}...")
        embeddings = load_flan_embeddings(file_path)
        rdm, rdm_up_tri = compute_rdm(embeddings)
        # Use the file name (without extension) as prefix
        prefix = os.path.splitext(os.path.basename(file_path))[0]
        save_rdms(rdm, rdm_up_tri, output_dir, prefix)
        print(f"Saved RDMs for {prefix} (shape: {rdm.shape})")

if __name__ == "__main__":
    main() 