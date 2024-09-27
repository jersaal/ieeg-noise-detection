import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import toml
from utils import VAE, compute_eeg_spectra, add_simulated_noise

# Load general configuration from the TOML file
config_fn = 'config.toml'
config = toml.load(config_fn)

np.random.seed(42)  # Set your preferred seed
torch.manual_seed(42)  # For CPU
torch.cuda.manual_seed(42)  # For current GPU
torch.cuda.manual_seed_all(42)  # If you use multiple GPUs

# Extract configurations from the TOML file
RCS0X = config['general']['vae_participant_id']
num_channels = config['data']['channels']
num_trials = config['data']['num_trials']
val_num_trials = config['data']['val_num_trials']
samples_per_epoch = int(config['data']['duration'] * config['data']['sampling_rate'])
sr = config['data']['sampling_rate']
latent_factors = config['vae']['latent_factors']
noise_amplitude = np.array(config['noise']['amplitudes'])
noise_frequencies = np.append(np.arange(5, 141, 5), 0.5)

# Directories
base_dir = config['general']['base_dir']
traces_dir = os.path.join(base_dir, 'traces/')
raw_noisy_dir = os.path.join(traces_dir, 'raw_noisy/')
clean_dir = os.path.join(traces_dir, 'clean/')
spectra_dir = os.path.join(base_dir, 'spectra/')
spectra_fig_dir = os.path.join(spectra_dir, 'spectra_figures/')
tr_clean_spectra_dir = os.path.join(spectra_dir, 'clean/')
tr_noisy_spectra_dir = os.path.join(spectra_dir, 'noisy/')
noisy_spec_fig_dir = os.path.join(spectra_fig_dir, 'noisy/')
clean_spec_fig_dir = os.path.join(spectra_fig_dir, 'clean/')

# Create necessary directories
os.makedirs(raw_noisy_dir, exist_ok=True)
os.makedirs(clean_dir, exist_ok=True)
os.makedirs(tr_clean_spectra_dir, exist_ok=True)
os.makedirs(tr_noisy_spectra_dir, exist_ok=True)
os.makedirs(noisy_spec_fig_dir, exist_ok=True)
os.makedirs(clean_spec_fig_dir, exist_ok=True)

# Directories for validation
val_traces_dir = os.path.join(base_dir, 'val_traces/')
val_raw_noisy_dir = os.path.join(val_traces_dir, 'raw_noisy/')
val_clean_dir = os.path.join(val_traces_dir, 'clean/')
val_spectra_dir = os.path.join(base_dir, 'val_spectra/')
val_spectra_fig_dir = os.path.join(val_spectra_dir, 'spectra_figures/')
val_tr_clean_spectra_dir = os.path.join(val_spectra_dir, 'clean/')
val_tr_noisy_spectra_dir = os.path.join(val_spectra_dir, 'noisy/')
val_noisy_spec_fig_dir = os.path.join(val_spectra_fig_dir, 'noisy/')
val_clean_spec_fig_dir = os.path.join(val_spectra_fig_dir, 'clean/')

# Create necessary directories
os.makedirs(val_raw_noisy_dir, exist_ok=True)
os.makedirs(val_clean_dir, exist_ok=True)
os.makedirs(val_tr_clean_spectra_dir, exist_ok=True)
os.makedirs(val_tr_noisy_spectra_dir, exist_ok=True)
os.makedirs(val_noisy_spec_fig_dir, exist_ok=True)
os.makedirs(val_clean_spec_fig_dir, exist_ok=True)

# Set up the device for PyTorch (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_synthetic_eeg_epoch(model, num_samples):
    """
    Generate a synthetic EEG epoch using the trained VAE model.

    Parameters:
    - model (torch.nn.Module): The trained VAE model.
    - num_samples (int): Number of samples in the epoch.

    Returns:
    - numpy.ndarray: Generated synthetic EEG epoch.
    """
    with torch.no_grad():
        latent_sample = torch.randn(1, latent_factors).to(device)
        epoch = model.decoder(latent_sample).cpu().numpy().squeeze()
    return epoch

def generate_data(sel_num_trials, sel_clean_dir, sel_raw_noisy_dir):
    """
    Generate synthetic clean and raw noisy EEG data for all channels.

    Parameters:
    - sel_num_trials (int): Number of trials to generate.
    - sel_clean_dir (str): Directory to save clean EEG data.
    - sel_raw_noisy_dir (str): Directory to save raw noisy EEG data.

    Returns:
    - None
    """

    for channel in range(num_channels):
        print(f"Processing channel {channel}...")

        model_path = os.path.join(config['general']['output_dir'], f'{RCS0X}_vae_channel_{channel}.pth')

        model = VAE(data_len=samples_per_epoch).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        clean_trials = np.zeros((sel_num_trials, samples_per_epoch))
        raw_noisy_trials = np.zeros((sel_num_trials, samples_per_epoch))

        for trial_idx in range(sel_num_trials):
            print(f"Generating trial {trial_idx + 1}/{sel_num_trials} for channel {channel}...")
            clean_trials[trial_idx] = generate_synthetic_eeg_epoch(model, samples_per_epoch)
            raw_noisy_trials[trial_idx] = generate_synthetic_eeg_epoch(model, samples_per_epoch)

        np.save(os.path.join(sel_clean_dir, f'channel_{channel}.npy'), clean_trials)
        np.save(os.path.join(sel_raw_noisy_dir, f'channel_{channel}.npy'), raw_noisy_trials)

    print("Per-channel clean and raw noisy data saved successfully.")


def organize_into_trials(sel_clean_dir, sel_raw_noisy_dir, sel_num_trials):
    """
    Organize per-channel data into trial files containing data from all channels.

    Parameters:
    - sel_clean_dir (str): Directory containing clean per-channel data.
    - sel_raw_noisy_dir (str): Directory containing raw noisy per-channel data.
    - sel_num_trials (int): Number of trials to organize.

    Returns:
    - None
    """
    def load_and_organize_trials(num_channels, sel_num_trials, data_dir, data_type):
        trials_data = np.zeros((sel_num_trials, num_channels, samples_per_epoch))
        for channel in range(num_channels):
            data = np.load(os.path.join(data_dir, f'channel_{channel}.npy'))
            trials_data[:, channel, :] = data[:sel_num_trials]
        for trial_idx in range(sel_num_trials):
            np.save(os.path.join(data_dir, f'trial_{trial_idx}.npy'), trials_data[trial_idx])

    print("Organizing clean data into trial files...")
    load_and_organize_trials(num_channels, sel_num_trials, sel_clean_dir, 'clean')
    print("Organizing raw noisy data into trial files...")
    load_and_organize_trials(num_channels, sel_num_trials, sel_raw_noisy_dir, 'raw_noisy')
    print("Data organized into trial files successfully.")


def add_noise(sel_traces_dir, sel_num_trials):
    """
    Add simulated noise to the raw noisy EEG data to create noisy trials.

    Parameters:
    - sel_traces_dir (str): Directory containing raw noisy trials.
    - sel_num_trials (int): Number of trials to process.

    Returns:
    - None
    """
    raw_noisy_trial_dir = os.path.join(sel_traces_dir, 'raw_noisy')
    noisy_trial_dir = os.path.join(sel_traces_dir, 'noisy')
    os.makedirs(noisy_trial_dir, exist_ok=True)

    for trial_idx in range(sel_num_trials):
        print(f"Processing trial {trial_idx}...")

        raw_noisy_trials = np.load(os.path.join(raw_noisy_trial_dir, f'trial_{trial_idx}.npy'))
        noisy_trials = raw_noisy_trials.copy()
        base_amplitude = np.random.choice(noise_amplitude)
        base_frequency = np.random.choice(noise_frequencies)
        num_harmonics = np.random.randint(1, 15)
        channels_with_noise = np.random.choice(num_channels, size=np.random.randint(3, 11), replace=False)
        print(f"Base frequency: {base_frequency} Hz, Number of harmonics: {num_harmonics}")

        for channel in channels_with_noise:
            noisy_epoch = noisy_trials[channel].copy()
            for harmonic in range(1, num_harmonics + 1):
                harmonic_frequency = base_frequency * harmonic
                noisy_epoch = add_simulated_noise(noisy_epoch, sr, harmonic_frequency, base_amplitude)
            noisy_trials[channel] = noisy_epoch

        np.save(os.path.join(noisy_trial_dir, f'trial_{trial_idx}.npy'), noisy_trials)

    print("Noise added to trials successfully.")


def generate_spectra(sel_traces_dir, sel_tr_clean_spectra_dir, sel_tr_noisy_spectra_dir, sel_num_trials):
    """
    Compute and save the power spectral densities (PSDs) of clean and noisy trials.

    Parameters:
    - sel_traces_dir (str): Directory containing trial data.
    - sel_tr_clean_spectra_dir (str): Directory to save clean spectra.
    - sel_tr_noisy_spectra_dir (str): Directory to save noisy spectra.
    - sel_num_trials (int): Number of trials to process.

    Returns:
    - None
    """
    for trial_idx in range(sel_num_trials):
        print(f"Processing trial {trial_idx}...")

        clean_trials = np.load(os.path.join(sel_traces_dir, 'clean', f'trial_{trial_idx}.npy'))
        freqs, clean_spectra = compute_eeg_spectra(clean_trials, sr)
        np.savez(os.path.join(sel_tr_clean_spectra_dir, f'trial_{trial_idx}.npz'), freqs=freqs, spectra=clean_spectra)

        noisy_trials = np.load(os.path.join(sel_traces_dir, 'noisy', f'trial_{trial_idx}.npy'))
        freqs, noisy_spectra = compute_eeg_spectra(noisy_trials, sr)
        np.savez(os.path.join(sel_tr_noisy_spectra_dir, f'trial_{trial_idx}.npz'), freqs=freqs, spectra=noisy_spectra)

    print("Spectra data saved successfully.")


def plot_eeg_spectra_batch(filenames, image_fps):
    """
    Plot and save EEG spectra images for a batch of trials.

    Parameters:
    - filenames (list of str): List of filenames containing the EEG spectra data.
    - image_fps (list of str): List of file paths to save the images.

    Returns:
    - None
    """
    for filename, image_fp in zip(filenames, image_fps):
        data = np.load(filename, allow_pickle=True)
        freqs = data['freqs']
        spectra = data['spectra']
        spectra = spectra[:, freqs <= 150]
        freqs = freqs[freqs <= 150]
        plt.figure(figsize=(10, 6))
        for channel in range(spectra.shape[0]):
            plt.plot(freqs, np.log10(spectra[channel]), color='black', alpha=0.5, lw=0.5)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.axis('off')
        plt.savefig(image_fp, bbox_inches='tight', pad_inches=0, dpi=100)
        #also save pdf version
        pdf_fp = image_fp.replace('.png', '.pdf')
        plt.close()
        print(f"Saved image to {image_fp}")


def process_trials(data_dir, fig_dir, trial_type, batch_size=10):
    """
    Process trials and save spectra images for clean and noisy data.

    Parameters:
    - data_dir (str): Directory containing the trial spectra data files.
    - fig_dir (str): Directory to save the resulting images.
    - trial_type (str): Type of trial ("clean" or "noisy").
    - batch_size (int, optional): Number of trials to process in each batch. Default is 10.

    Returns:
    - None
    """
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        image_fps = [os.path.join(fig_dir, f'{trial_type}_trial_{j}.png') for j in range(i, i + len(batch_files))]
        print(f"Processing batch {i // batch_size + 1}...")
        plot_eeg_spectra_batch(batch_files, image_fps)


def gen_main():
    """
    Main function to generate synthetic EEG data, add noise, and save spectra images.

    Executes the entire pipeline:
    - Generates synthetic data.
    - Organizes data into trials.
    - Adds simulated noise.
    - Computes and saves spectra.
    - Plots and saves spectra images.

    Returns:
    - None
    """
    generate_data(num_trials, clean_dir, raw_noisy_dir)
    organize_into_trials(clean_dir, raw_noisy_dir, num_trials)
    add_noise(traces_dir, num_trials)
    generate_spectra(traces_dir, tr_clean_spectra_dir, tr_noisy_spectra_dir, num_trials)

    # Plot and save spectra images
    process_trials(tr_noisy_spectra_dir, noisy_spec_fig_dir, 'noisy')
    process_trials(tr_clean_spectra_dir, clean_spec_fig_dir, 'clean')

if __name__ == "__main__":
    gen_main()
