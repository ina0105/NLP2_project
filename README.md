# NLP2 Project

This repository contains code for processing and analyzing fMRI data, with a focus on language and visual network activations. The project processes fMRI data from multiple participants and generates Representational Dissimilarity Matrices (RDMs) for both language and visual networks.

## Project Structure

```
NLP2_project/
├── LLM/                  # Language Model related code
│   ├── generate_rdms/    # RDM generation scripts
│   ├── one_word/        # Single word experiments
│   ├── one_general/     # Single general context experiments
│   ├── one_distinct/    # Single distinct context experiments
│   ├── five_general/    # Five general context experiments
│   ├── five_distinct/   # Five distinct context experiments
│   └── compute_rsa/     # RSA computation scripts
├── VLM/                  # Vision Language Model related code
│   ├── rsa_results_vlm/ # VLM RSA results
│   └── [various scripts]# VLM processing and analysis scripts
├── rdm_output/          # Output directory for RDMs
│   ├── languageLH/      # Language network RDMs
│   └── visual/          # Visual network RDMs
├── process_fmri.py      # Main script for processing fMRI data
├── environment.yml      # Conda environment configuration
└── requirements.txt     # Python package requirements
```

## Requirements

The project requires Python 3.10.12 and several key dependencies. You can set up the environment using either:

### Using Conda (Recommended)
```bash
conda env create -f environment.yml
```

### Using pip
```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch 2.0.1
- NumPy
- FAISS-CPU 1.7.4
- Transformers 4.30.0
- Datasets 2.16.1
- SentencePiece
- And other scientific computing packages

## fMRI Data Processing

The main script `process_fmri.py` processes fMRI data from multiple participants and generates RDMs for both language and visual networks. The script:

1. Loads fMRI data for each participant
2. Extracts network activations for language and visual networks
3. Computes RDMs using cosine distances

## Language Model (LLM) Processing

The `LLM` directory contains scripts for processing and analyzing language model representations:

### Directory Structure
- `generate_rdms/`: Scripts for generating RDMs from language model representations
- `one_word/`: Experiments with single word contexts
- `one_general/`: Experiments with single general context
- `one_distinct/`: Experiments with single distinct context
- `five_general/`: Experiments with five general contexts
- `five_distinct/`: Experiments with five distinct contexts
- `compute_rsa/`: Scripts for computing Representational Similarity Analysis

### Usage
1. Navigate to the appropriate experiment directory
2. Run the corresponding generation script
3. Use the RSA computation scripts to analyze the results

## Vision Language Model (VLM) Processing

The `VLM` directory contains scripts for processing and analyzing vision-language model representations using BLIP-2:

### Key Components
- `extract_blip2_representations.py`: Contains the general helper functions for extracting representations from the BLIP-2 model
- `make_vlm_rdm.py`: Generates RDMs from VLM representations
- `compute_vlm_rsa.py`: Computes RSA between VLM and fMRI representations
- Various generation scripts for different context types:
  - `generate_blip2_one_word.py`
  - `generate_blip2_one_general.py`
  - `generate_blip2_one_distinct.py`
  - `generate_blip2_five_general.py`
  - `generate_blip2_five_distinct.py`

### Usage
1. Ensure you have the VLM environment set up:
```bash
conda env create -f VLM/vlm_environment.yml
```

2. Extract representations (using the `One Word` context type as example):
```bash
python generate_blip2_one_word.py
```

3. Generate RDMs:
```bash
python make_vlm_rdm.py
```

4. Compute RSA:
```bash
python compute_vlm_rsa.py
```

## Output Format

The RDMs are saved in the `rdm_output` directory, organized by network type:
- `languageLH/`: Contains language network RDMs
- `visual/`: Contains visual network RDMs

For each participant and network, the following files are generated:
- Full RDM matrix
- Upper triangle of the RDM matrix
- Multiple file formats for compatibility

## Usage

To process the fMRI data:

```bash
python process_fmri.py
```

The script will automatically create the necessary output directories and process the data for all participants.

## Contact

