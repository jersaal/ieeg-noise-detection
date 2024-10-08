# Robust Detection of Brain Stimulation Artifacts in iEEG

This repository contains the code accompanying the manuscript:

**"Robust Detection of Brain Stimulation Artifacts in iEEG Using Autoencoder-Generated Signals and ResNet Classification"**
https://www.biorxiv.org/content/10.1101/2024.09.30.615930v1

## Overview

This project implements a pipeline for detecting stimulation-induced noise in intracranial EEG (iEEG) recordings using Variational Autoencoders (VAEs) and a ResNet-18 classifier.

## Repository Structure

- `scripts/` Contains scripts used for all analyses in the manuscript.
  - `vae_pipeline.py`: Train VAEs on real iEEG data and generate synthetic clean signals.
  - `generate_noisy_spectra.py`: Add simulated noise to synthetic data and generate spectra images.
  - `train_resnet.py`: Train the ResNet-18 classifier on synthetic data.
  - `validate_real_data.py`: Validate the trained classifier on real iEEG data.
  - `vae_autocorr_analysis.py`: Analyze autocorrelation of real and synthetic EEG data.
  - `utils.py`: Contains shared classes and functions used across scripts.
  - `config.toml`: Configuration file with parameters and file paths for full analysis. 
  - `RCS04_vae_selected_fns.toml`: Example file with EDF filenames to be used for VAE training.
  - `classify_config.toml`: Configuration file with parameters and file paths for classification given a pre-trained ResNet model.
- `data/`
  - `resnet_model.pth`: ResNet model trained on VAE-generated data.
- `classification/` Contains scripts for classifying iEEG data as noisy or clean using the pre-trained ResNet model.
    - `classify_ieeg.py`: Classify 2.5s segments of iEEG data using classify_config.toml.
- `README.md`: Project documentation.


## How to Use

The following instructions allow users to classify a set of 2.5s .fif files containing clean and noisy iEEG signals. 

1. **Setup the Configuration File**:
   - The script uses a configuration file (`classify_config.toml`) to specify file paths and decision threshold.   
   - Modify the `classify_config.toml` file to point to the correct locations for your data and pre-trained model. An example configuration file is provided.

2. **Run the Classification Script**:
   - Once the configuration file is set up, make sure you are in the root directory of the project. You can run the classification script with the following command:
   python classification/classify_ieeg.py

3. **Output**:
   - The script will output a CSV file with the classification results for each input file.
   - The script will process all .fif files in the specified data folder and classify each file as "noisy" or "clean". It will save the results in a CSV file (classification_results.csv) located in the output folder.

   - The CSV file will contain the following columns:
        - filename: Name of the EEG file.
        - noise: A boolean value (True for noisy, False for clean).
        - noise_prob: The model's predicted probability that the file contains noise.
