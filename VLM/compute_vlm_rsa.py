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

def compare_networks_by_condition(network_correlations):
    """
    Compare correlations between networks for each condition
    network_correlations: dict with network names as keys, each containing condition correlations
    """
    network_comparisons = []
    networks = list(network_correlations.keys())
    
    if len(networks) != 2:
        print("Warning: Expected exactly 2 networks for comparison")
        return network_comparisons
    
    net1, net2 = networks
    conditions = set(network_correlations[net1].keys()) & set(network_correlations[net2].keys())
    
    for condition in conditions:
        corr1 = network_correlations[net1][condition]
        corr2 = network_correlations[net2][condition]
        
        t_stat, p_val = ttest_rel(corr1, corr2)
        mean_diff = np.mean(corr1) - np.mean(corr2)
        
        network_comparisons.append({
            'condition': condition,
            'network1': net1,
            'network2': net2,
            't_statistic': t_stat,
            'p_value': p_val,
            'mean_diff': mean_diff,
            'mean_corr_net1': np.mean(corr1),
            'mean_corr_net2': np.mean(corr2)
        })
    
    return network_comparisons

def main():
    # --- BLIP2 VLM RDMs ---
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

    # Output directories
    os.makedirs('rsa_results_vlm/statistical_tests', exist_ok=True)

    # Store all correlations for cross-network comparison
    all_network_correlations = {}

    for network, rdm_files in brain_networks.items():
        if not rdm_files:
            print(f"No RDM files found for network: {network}")
            continue

        condition_correlations = {}

        for label, vlm_file in vlm_rdm_files.items():
            if not os.path.exists(vlm_file):
                print(f"BLIP2 RDM not found: {vlm_file}")
                continue

            vlm_rdm = load_rdm(vlm_file)
            correlations = compute_participant_level_correlations(vlm_rdm, rdm_files)
            condition_correlations[label] = correlations

            # Save participant-level correlations
            np.save(f'rsa_results_vlm/statistical_tests/{network}_{label}_correlations.npy', correlations)

        all_network_correlations[network] = condition_correlations

        # Perform paired t-tests within network
        comparisons = compare_conditions(condition_correlations)

        # Save t-test results
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

    # NEW: Cross-network comparisons
    if len(all_network_correlations) == 2:
        network_comparisons = compare_networks_by_condition(all_network_correlations)
        
        # Save cross-network comparison results
        with open('rsa_results_vlm/statistical_tests/network_comparisons.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['condition', 'network1', 'network2', 't_statistic', 'p_value', 'mean_diff', 'mean_corr_net1', 'mean_corr_net2'])
            writer.writeheader()
            for comp in network_comparisons:
                writer.writerow(comp)

        # Print cross-network comparison results
        print(f"\nCross-network comparisons (BLIP-2):")
        print("======================================================")
        for comp in network_comparisons:
            significance = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else "ns"
            print(f"Condition: {comp['condition']}")
            print(f"  {comp['network1']}: {comp['mean_corr_net1']:.3f}")
            print(f"  {comp['network2']}: {comp['mean_corr_net2']:.3f}")
            print(f"  Mean difference: {comp['mean_diff']:.3f}")
            print(f"  t-statistic: {comp['t_statistic']:.3f}")
            print(f"  p-value: {comp['p_value']:.3e} {significance}")
            print("------------------------------------------------------")

if __name__ == "__main__":
    main()