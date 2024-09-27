# Robust Detection of Brain Stimulation Artifacts in iEEG

This repository contains the code accompanying the manuscript:

**"Robust Detection of Brain Stimulation Artifacts in iEEG Using Autoencoder-Generated Signals and ResNet Classification"**

## Overview

This project implements a pipeline for detecting stimulation-induced noise in intracranial EEG (iEEG) recordings using Variational Autoencoders (VAEs) and a ResNet-18 classifier.

## Repository Structure

- `scripts/`
  - `vae_pipeline.py`: Train VAEs on real iEEG data and generate synthetic clean signals.
  - `generate_noisy_spectra.py`: Add simulated noise to synthetic data and generate spectra images.
  - `train_resnet.py`: Train the ResNet-18 classifier on synthetic data.
  - `validate_real_data.py`: Validate the trained classifier on real iEEG data.
  - `vae_autocorr_analysis.py`: Analyze autocorrelation of real and synthetic EEG data.
  - `utils.py`: Contains shared classes and functions used across scripts.
  - `config.toml`: Configuration file with parameters and file paths.
  - `RCS04_vae_selected_fns.toml`: Example file with selected VAEs for a specific patient.
- `data/`
  - `resnet_model.pth`: Examp
- `README.md`: Project documentation.
- `LICENSE`: Project license.
