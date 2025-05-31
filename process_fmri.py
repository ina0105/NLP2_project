import torch
import numpy as np
import json
from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist, squareform
import os
participants={'01':'/scratch-shared/scur2185/M01/data_180concepts_pictures.mat', '02':'/scratch-shared/scur2185/M02/data_180concepts_pictures.mat',
               '03':'/scratch-shared/scur2185/M03/data_180concepts_pictures.mat', '04':'/scratch-shared/scur2185/M04/data_180concepts_pictures.mat', 
               '05':'/scratch-shared/scur2185/M05/data_180concepts_pictures.mat', '06':'/scratch-shared/scur2185/M06/data_180concepts_pictures.mat', 
               '07':'/scratch-shared/scur2185/M07/data_180concepts_pictures.mat', '08':'/scratch-shared/scur2185/M08/data_180concepts_pictures.mat',
                '09':'/scratch-shared/scur2185/M09/data_180concepts_pictures.mat', '10':'/scratch-shared/scur2185/M10/data_180concepts_pictures.mat',
                '13':'/scratch-shared/scur2185/M13/data_180concepts_pictures.mat', '14':'/scratch-shared/scur2185/M14/data_180concepts_pictures.mat',
                '15':'/scratch-shared/scur2185/M15/data_180concepts_pictures.mat', '16':'/scratch-shared/scur2185/M16/data_180concepts_pictures.mat',
                '17':'/scratch-shared/scur2185/M17/data_180concepts_pictures.mat'}
os.makedirs('rdm_output/languageLH', exist_ok=True)
os.makedirs('rdm_output/visual', exist_ok=True)
def get_network_activations(fmri_data_matlab, network_name):

  networks = [atlas[0] for atlas in fmri_data_matlab['meta']['atlases'][0][0][0]]
  nw_index = networks.index(network_name)

  nw_columns = fmri_data_matlab['meta']['roiColumns'][0][0][0][nw_index]
  column_indexes = np.concat([nw_columns[roi][0].flatten()-1 for roi in range(len(nw_columns))], axis=0)

  network_responses = fmri_data_matlab['examples'][:, column_indexes]

  return network_responses
for participant, file in participants.items():
    # Load the fMRI data for the participant
    fmri_data = loadmat(file)
    # Get the network activations for the languageLH network
    langLH_fmri_data = get_network_activations(fmri_data, 'languageLH')
    # Obtain the RDM with cosine distances
    rdm = squareform(pdist(langLH_fmri_data))
    # Select values from the upper triangle
    rdm_up_tri = rdm[np.triu_indices(n=180, k=1, m=180)] # k=1 means that we don't want to include values on the diagonal
    # Save as NumPy file (.npy) - most efficient for Python reuse
    np.save(f'rdm_output/languageLH/participant_{participant}_langLH_rdm.npy', rdm)
    np.save(f'rdm_output/languageLH/participant_{participant}_langLH_rdm_uptri.npy', rdm_up_tri)
    
    # Save as CSV file - good for universal compatibility
    np.savetxt(f'rdm_output/languageLH/participant_{participant}_langLH_rdm.csv', rdm, delimiter=',')
    np.savetxt(f'rdm_output/languageLH/participant_{participant}_langLH_rdm_uptri.csv', rdm_up_tri, delimiter=',')
    
    # Save as MATLAB file (.mat) - good for MATLAB users
    savemat(f'rdm_output/languageLH/participant_{participant}_langLH_rdm.mat', {'rdm': rdm, 'rdm_up_tri': rdm_up_tri})
    
    # Save as JSON file - good for web applications
    with open(f'rdm_output/languageLH/participant_{participant}_langLH_rdm.json', 'w') as f:
        json.dump(rdm.tolist(), f)
    with open(f'rdm_output/languageLH/participant_{participant}_langLH_rdm_uptri.json', 'w') as f:
        json.dump(rdm_up_tri.tolist(), f)
    print(f"Participant {participant}: RDM shape: {rdm_up_tri.shape}")

for participant, file in participants.items():
    # Load the fMRI data for the participant
    fmri_data = loadmat(file)
    # Get the network activations for the visual network
    visual_fmri_data = get_network_activations(fmri_data, 'visual')
    # Obtain the RDM with cosine distances
    rdm = squareform(pdist(visual_fmri_data))
    # Select values from the upper triangle
    rdm_up_tri = rdm[np.triu_indices(n=180, k=1, m=180)] # k=1 means that we don't want to include values on the diagonal
    np.save(f'rdm_output/visual/participant_{participant}_visual_rdm.npy', rdm)
    np.save(f'rdm_output/visual/participant_{participant}_visual_rdm_uptri.npy', rdm_up_tri)
    
    # Save as CSV file
    np.savetxt(f'rdm_output/visual/participant_{participant}_visual_rdm.csv', rdm, delimiter=',')
    np.savetxt(f'rdm_output/visual/participant_{participant}_visual_rdm_uptri.csv', rdm_up_tri, delimiter=',')
    
    # Save as MATLAB file (.mat)
    savemat(f'rdm_output/visual/participant_{participant}_visual_rdm.mat', {'rdm': rdm, 'rdm_up_tri': rdm_up_tri})
    
    # Save as JSON file
    with open(f'rdm_output/visual/participant_{participant}_visual_rdm.json', 'w') as f:
        json.dump(rdm.tolist(), f)
    with open(f'rdm_output/visual/participant_{participant}_visual_rdm_uptri.json', 'w') as f:
        json.dump(rdm_up_tri.tolist(), f)
    print(f"Participant {participant}: RDM shape: {rdm_up_tri.shape}")
