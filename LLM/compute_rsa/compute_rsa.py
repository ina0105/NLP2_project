import numpy as np
import os
from scipy.stats import spearmanr
import json
import glob
import csv

def load_rdm(file_path):
    """Load RDM from various file formats"""
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.csv'):
        return np.loadtxt(file_path, delimiter=',')
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return np.array(json.load(f))
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def compute_rsa(rdm1, rdm2):
    """Compute RSA using Spearman correlation between upper triangles"""
    n = rdm1.shape[0]
    rdm1_up_tri = rdm1[np.triu_indices(n=n, k=1)]
    rdm2_up_tri = rdm2[np.triu_indices(n=n, k=1)]
    
    correlation, p_value = spearmanr(rdm1_up_tri, rdm2_up_tri)
    
    return correlation, p_value

def average_rdms(rdm_files):
    rdms = [load_rdm(f) for f in rdm_files]
    return np.mean(rdms, axis=0)

def main():
    flan_rdm_files = [
        'rdm_output/llm/flan_t5_one_word_rdm.npy',
        'rdm_output/llm/flan_t5_one_general_rdm.npy',
        'rdm_output/llm/flan_t5_one_distinct_rdm.npy',
        'rdm_output/llm/flan_t5_five_general_rdm.npy',
        'rdm_output/llm/flan_t5_five_distinct_rdm.npy',
    ]
    flan_rdm_labels = [
        'one_word', 'one_general', 'one_distinct', 'five_general', 'five_distinct'
    ]
    brain_networks = {
        'languageLH': glob.glob('rdm_output/languageLH/participant_*_langLH_rdm.npy'),
        'visual': glob.glob('rdm_output/visual/participant_*_visual_rdm.npy'),
    }
    results = []
    for flan_file, flan_label in zip(flan_rdm_files, flan_rdm_labels):
        if not os.path.exists(flan_file):
            print(f"FLAN RDM not found: {flan_file}")
            continue
        flan_rdm = load_rdm(flan_file)
        for network, rdm_files in brain_networks.items():
            if not rdm_files:
                print(f"No RDM files found for network: {network}")
                continue
            avg_brain_rdm = average_rdms(rdm_files)
            correlation, p_value = compute_rsa(flan_rdm, avg_brain_rdm)
            results.append({
                'flan_rdm': flan_label,
                'brain_network': network,
                'correlation': correlation,
                'p_value': p_value
            })
    os.makedirs('rsa_results', exist_ok=True)
    with open('rsa_results/rsa_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open('rsa_results/rsa_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['flan_rdm', 'brain_network', 'correlation', 'p_value'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print("\nFLAN RDM\tBrain Network\tCorrelation\tP-value")
    print("------------------------------------------------------")
    for row in results:
        print(f"{row['flan_rdm']}\t{row['brain_network']}\t{row['correlation']:.3f}\t{row['p_value']:.3e}")

if __name__ == "__main__":
    main() 