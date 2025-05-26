import numpy as np
import os
from scipy.stats import spearmanr, ttest_rel
import json
import glob
import csv
from itertools import combinations

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

def compute_participant_level_correlations(vlm_rdm, brain_rdm_files):
    correlations = []
    for brain_file in brain_rdm_files:
        brain_rdm = load_rdm(brain_file)
        corr, _ = compute_rsa(vlm_rdm, brain_rdm)
        correlations.append(corr)
    return np.array(correlations)

def compare_conditions(correlations_dict):
    comparisons = []
    for (cond1, corr1), (cond2, corr2) in combinations(correlations_dict.items(), 2):
        t_stat, p_val = ttest_rel(corr1, corr2)
        mean_diff = np.mean(corr1) - np.mean(corr2)
        comparisons.append({
            'condition1': cond1,
            'condition2': cond2,
            't_statistic': t_stat,
            'p_value': p_val,
            'mean_diff': mean_diff,
            'mean_corr1': np.mean(corr1),
            'mean_corr2': np.mean(corr2)
        })
    return comparisons

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

    os.makedirs('rsa_results_vlm/statistical_tests', exist_ok=True)

    for network, rdm_files in brain_networks.items():
        if not rdm_files:
            print(f"No RDMs found for brain network: {network}")
            continue

        condition_correlations = {}

        for label, vlm_file in vlm_rdm_files.items():
            if not os.path.exists(vlm_file):
                print(f"VLM RDM not found: {vlm_file}")
                continue

            vlm_rdm = load_rdm(vlm_file)
            correlations = compute_participant_level_correlations(vlm_rdm, rdm_files)
            condition_correlations[label] = correlations

            # Save participant-level correlations
            np.save(f'rsa_results_vlm/statistical_tests/{network}_{label}_correlations.npy', correlations)

        # Perform condition comparisons
        comparisons = compare_conditions(condition_correlations)

        # Save to CSV
        with open(f'rsa_results_vlm/statistical_tests/{network}_comparisons.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['condition1', 'condition2', 't_statistic', 'p_value', 'mean_diff', 'mean_corr1', 'mean_corr2'])
            writer.writeheader()
            for comp in comparisons:
                writer.writerow(comp)

        # Print results
        print(f"\nStatistical comparisons for {network} network:")
        print("------------------------------------------------------")
        for comp in comparisons:
            print(f"{comp['condition1']} vs {comp['condition2']}:")
            print(f"  Mean correlation {comp['condition1']}: {comp['mean_corr1']:.3f}")
            print(f"  Mean correlation {comp['condition2']}: {comp['mean_corr2']:.3f}")
            print(f"  Mean difference: {comp['mean_diff']:.3f}")
            print(f"  t-statistic: {comp['t_statistic']:.3f}")
            print(f"  p-value: {comp['p_value']:.3e}")
            print("------------------------------------------------------")

if __name__ == "__main__":
    main()
