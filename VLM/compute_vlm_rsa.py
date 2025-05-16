import numpy as np
import os
from scipy.stats import spearmanr
import json
import glob
import csv

def load_rdm(file_path):
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
    n = rdm1.shape[0]
    rdm1_ut = rdm1[np.triu_indices(n=n, k=1)]
    rdm2_ut = rdm2[np.triu_indices(n=n, k=1)]
    correlation, p_value = spearmanr(rdm1_ut, rdm2_ut)
    return correlation, p_value

def average_rdms(rdm_files):
    rdms = [load_rdm(f) for f in rdm_files]
    return np.mean(rdms, axis=0)

def main():
    # --- VLM RDMs ---
    vlm_rdm_dir = "rdm_output_blip2"
    vlm_rdm_files = {
        'one_word': os.path.join(vlm_rdm_dir, 'vlm_one_word_rdm.npy'),
        'one_general': os.path.join(vlm_rdm_dir, 'vlm_one_general_rdm.npy'),
        'one_distinct': os.path.join(vlm_rdm_dir, 'vlm_one_distinct_rdm.npy'),
        'five_general': os.path.join(vlm_rdm_dir, 'vlm_five_general_rdm.npy'),
        'five_distinct': os.path.join(vlm_rdm_dir, 'vlm_five_distinct_rdm.npy'),
    }

    # --- Brain RDMs ---
    brain_rdm_base = "/home/scur2169/NLP2_project/rdm_output"
    brain_networks = {
        'languageLH': glob.glob(os.path.join(brain_rdm_base, 'languageLH', 'participant_*_langLH_rdm.npy')),
        'visual': glob.glob(os.path.join(brain_rdm_base, 'visual', 'participant_*_visual_rdm.npy')),
    }

    results = []

    for vlm_label, vlm_path in vlm_rdm_files.items():
        if not os.path.exists(vlm_path):
            print(f"VLM RDM not found: {vlm_path}")
            continue
        vlm_rdm = load_rdm(vlm_path)

        for network, rdm_paths in brain_networks.items():
            if not rdm_paths:
                print(f"No RDMs found for brain network: {network}")
                continue
            avg_brain_rdm = average_rdms(rdm_paths)
            correlation, p_value = compute_rsa(vlm_rdm, avg_brain_rdm)

            results.append({
                'vlm_rdm': vlm_label,
                'brain_network': network,
                'correlation': correlation,
                'p_value': p_value
            })

    # --- Save Results ---
    os.makedirs('rsa_results_vlm', exist_ok=True)

    with open('rsa_results_vlm/rsa_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open('rsa_results_vlm/rsa_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['vlm_rdm', 'brain_network', 'correlation', 'p_value'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # --- Print Summary ---
    print("\nVLM RDM\t\tBrain Network\tCorrelation\tP-value")
    print("------------------------------------------------------")
    for row in results:
        print(f"{row['vlm_rdm']:<15}{row['brain_network']:<15}{row['correlation']:.3f}\t\t{row['p_value']:.3e}")

if __name__ == "__main__":
    main()
