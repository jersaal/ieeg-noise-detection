import numpy as np
import matplotlib.pyplot as plt
import torch
from statsmodels.tsa.stattools import acf
import os
import mne
from scipy.stats import pearsonr, combine_pvalues
import toml
import logging
from utils import VAE

# Load configuration from TOML file
config_fn = 'config.toml'
config = toml.load(config_fn)

# Extract variables from the configuration
RCS0X = config['general']['vae_participant_id']
base_dir = config['general']['base_dir']
output_dir = config['general']['output_dir']
log_dir = config['general']['log_dir']
fig_fold = os.path.join(output_dir, 'autocorr_analysis_figs/')
os.makedirs(fig_fold, exist_ok=True)

latent_factors = config['vae']['latent_factors']
kernel_len = config['vae']['kernel_len']
pad_len = config['vae']['pad_len']
stride_len = config['vae']['stride_len']
data_len = int(config['data']['duration'] * config['data']['sampling_rate'])
channels = config['data']['channels']

participant_toml_file = config["fnames"]["ieeg_for_vae_fnames"]
participant_config = toml.load(participant_toml_file)
file_paths = participant_config['files']

# Setup logging
logging_fn = os.path.join(log_dir, 'autocorr_training_log.log')
logging.basicConfig(filename=logging_fn, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load single-channel EEG data
def load_single_channel_eeg_data(file_paths, duration=5, ch=0):
    """
    Load EEG data from multiple files for a single channel.

    Parameters:
    - file_paths (list of str): List of file paths to EEG data files in FIF format.
    - duration (int, optional): Duration of each window in seconds. Default is 5.
    - ch (int, optional): Index of the channel to load. Default is 0.

    Returns:
    - numpy.ndarray: Array of EEG data windows for the specified channel.
    """
    all_data_appended = []
    for path in file_paths:
        raw = mne.io.read_raw_fif(path, verbose=False)
        sfreq = raw.info['sfreq']
        total_samples = int(sfreq * duration)
        all_data = raw.get_data()
        data = all_data[ch, :]
        num_windows = data.shape[0] // total_samples
        for j in range(num_windows):
            start_sample = j * total_samples
            end_sample = start_sample + total_samples
            if end_sample <= data.shape[0]:
                window = data[start_sample:end_sample]
                all_data_appended.append(window)
    return np.array(all_data_appended)

# Function to calculate autocorrelation
def autocorr(x):
    """
    Calculate the autocorrelation function of a 1D array.

    Parameters:
    - x (numpy.ndarray): Input array.

    Returns:
    - numpy.ndarray: Autocorrelation function of the input array.
    """
    return acf(x, fft=True, nlags=len(x) - 1)

# Function to perform autocorrelation and correlation analysis for a single channel
def analyze_autocorrelation_and_correlation(channel_num, file_paths):
    """
    Analyze autocorrelation and correlation between real and synthetic EEG data for a channel.

    Parameters:
    - channel_num (int): Channel number to analyze.
    - file_paths (list of str): List of file paths to EEG data files.

    Returns:
    - dict: Dictionary containing analysis results for the channel.
    """
    model = VAE(data_len, kernel_len, pad_len, stride_len, latent_factors).to(device)
    model.load_state_dict(torch.load(os.path.join(output_dir, f"{RCS0X}_vae_channel_{channel_num}.pth")))

    #model_save_path = os.path.join(output_dir, f"{participant_id}_vae_channel_{ch}.pth")
    model.eval()

    # Load EEG data from the specified channel
    eeg_data = load_single_channel_eeg_data(file_paths, duration=config['data']['duration'], ch=channel_num)

    # Generate synthetic EEG data
    with torch.no_grad():
        sample = torch.randn(eeg_data.shape[0], latent_factors).to(device)  # Generate latent vectors
        synthetic_eeg = model.decoder(sample).cpu().numpy().squeeze()

    # Calculate autocorrelation for real and synthetic data
    acf_real = np.array([autocorr(trial) for trial in eeg_data])
    acf_synthetic = np.array([autocorr(trial) for trial in synthetic_eeg])

    # Plot average autocorrelation
    plt.figure(figsize=(10, 6))
    plt.plot(acf_real.mean(axis=0), label='Real sEEG')
    plt.plot(acf_synthetic.mean(axis=0), label='Synthetic sEEG')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.title(f'Autocorrelation Function Comparison - Channel {channel_num}')
    plt.xlim([0,500])
    plt.savefig(os.path.join(fig_fold, f'autocorr_comparison_channel_{channel_num}.png'))
    plt.savefig(os.path.join(fig_fold, f'autocorr_comparison_channel_{channel_num}.pdf'))
    plt.close()

    # Calculate a summary statistic: mean autocorrelation across all lags
    mean_acf_real = acf_real.mean()
    mean_acf_synthetic = acf_synthetic.mean()

    # Correlation analysis between the two autocorrelation curves
    correlation_coefficient, correlation_p_value = pearsonr(acf_real.mean(axis=0), acf_synthetic.mean(axis=0))

    # Log detailed results for this channel
    logging.info(f"Channel {channel_num}:")
    logging.info(f"  Mean Autocorrelation (Real): {mean_acf_real}")
    logging.info(f"  Mean Autocorrelation (Synthetic): {mean_acf_synthetic}")
    logging.info(f"  Correlation Coefficient: {correlation_coefficient}, p={correlation_p_value}")

    return {
        'channel': channel_num,
        'mean_acf_real': mean_acf_real,
        'mean_acf_synthetic': mean_acf_synthetic,
        'correlation_coefficient': correlation_coefficient,
        'correlation_p_value': correlation_p_value
    }

# Function to aggregate results and create summary plots
def aggregate_and_plot_results(results):
    """
    Aggregate analysis results across channels and create summary plots.

    Parameters:
    - results (list of dict): List of analysis results for each channel.

    Returns:
    - None
    """
    all_mean_acf_real = [r['mean_acf_real'] for r in results]
    all_mean_acf_synthetic = [r['mean_acf_synthetic'] for r in results]
    all_correlation_coefficients = [r['correlation_coefficient'] for r in results]
    all_correlation_p_values = [r['correlation_p_value'] for r in results]

    # Combine p-values across channels
    combined_stat, combined_p_value = combine_pvalues(all_correlation_p_values, method='fisher')

    # Print the range of Pearson correlation coefficients
    min_corr = min(all_correlation_coefficients)
    max_corr = max(all_correlation_coefficients)
    print(f"Range of Pearson correlation coefficients: {min_corr} to {max_corr}")

    # Plotting the distribution of mean autocorrelation values across channels
    plt.figure(figsize=(10, 6))
    plt.hist(all_mean_acf_real, bins=30, alpha=0.7, label='Real sEEG', color='blue')
    plt.hist(all_mean_acf_synthetic, bins=30, alpha=0.7, label='Synthetic sEEG', color='orange')
    plt.xlim(0, 1)
    plt.xlabel('Mean Autocorrelation')
    plt.ylabel('Number of Channels')
    plt.legend()
    plt.title('Distribution of Mean Autocorrelation Across Channels')
    plt.savefig(os.path.join(fig_fold, 'mean_acf_distribution.png'))
    plt.savefig(os.path.join(fig_fold, 'mean_acf_distribution.pdf'))
    plt.close()

    # Plotting the distribution of correlation coefficients across channels with various stylings
    # Basic Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_correlation_coefficients, bins=20, alpha=0.7, color='green')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Number of Channels')
    plt.xlim(0, 1)
    plt.title('Correlation Coefficient Distribution')
    plt.savefig(os.path.join(fig_fold, 'correlation_coefficient_distribution.png'))
    plt.savefig(os.path.join(fig_fold, 'correlation_coefficient_distribution.pdf'))
    plt.close()

    # Histogram with bounds [0-1]
    bounded_correlations = np.clip(all_correlation_coefficients, 0, 1)
    plt.figure(figsize=(10, 6))
    plt.hist(bounded_correlations, bins=18, alpha=0.7, color='purple')
    plt.xlabel('Correlation Coefficient (Bounded 0-1)')
    plt.ylabel('Number of Channels')
    plt.title('Correlation Coefficients Bounded 0-1')
    plt.savefig(os.path.join(fig_fold, 'correlation_coefficient_bounded.png'))
    plt.savefig(os.path.join(fig_fold, 'correlation_coefficient_bounded.pdf'))
    plt.close()

    # Histogram with Grid and Custom Line Styles
    plt.figure(figsize=(10, 6))
    plt.hist(all_correlation_coefficients, bins=20, alpha=0.7, color='red', edgecolor='black')
    plt.xlim(0,1)
    plt.grid(True)
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Number of Channels')
    plt.title('Correlation Coefficient Distribution with Grid and Style')
    plt.savefig(os.path.join(fig_fold, 'correlation_coefficient_grid.png'))
    plt.savefig(os.path.join(fig_fold, 'correlation_coefficient_grid.pdf'))
    plt.close()

    # Plot average autocorrelation functions across all channels
    mean_acf_real_avg = np.mean(all_mean_acf_real)
    mean_acf_synthetic_avg = np.mean(all_mean_acf_synthetic)
    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(all_mean_acf_real), label='Average Real sEEG ACF')
    plt.plot(np.mean(all_mean_acf_synthetic), label='Average Synthetic sEEG ACF')
    plt.xlabel('Lag')
    plt.ylabel('Mean Autocorrelation')
    plt.legend()
    plt.title('Average Autocorrelation Functions Across Channels')
    plt.savefig(os.path.join(fig_fold, 'average_acf_comparison.png'))
    plt.savefig(os.path.join(fig_fold, 'average_acf_comparison.pdf'))
    plt.close()

    # Logging overall statistics
    logging.info(f"\nOverall Results:")
    logging.info(f"  Combined p-value from correlation across all channels (Fisher's method): {combined_p_value}")
    logging.info(f"  Average Mean Autocorrelation (Real): {mean_acf_real_avg}")
    logging.info(f"  Average Mean Autocorrelation (Synthetic): {mean_acf_synthetic_avg}")
    logging.info(f"  Number of Channels with Correlation Coefficient > 0.8: {sum(r > 0.8 for r in all_correlation_coefficients)} out of {len(results)}")
    logging.info(f"  Range of Pearson correlation coefficients: {min_corr} to {max_corr}")

# Main execution function
def main():
    """
    Main function to perform autocorrelation analysis for all channels.

    Executes the following steps:
    - Analyzes each channel individually.
    - Aggregates results across channels.
    - Plots summary statistics and distributions.

    Returns:
    - None
    """
    # Initialize list to store results
    results = []

    # Analyze all channels
    for channel_num in range(channels):
        print(channel_num)
        logging.info(f"Analyzing channel {channel_num}...")
        result = analyze_autocorrelation_and_correlation(channel_num, file_paths)
        results.append(result)

    # Aggregate results and plot summaries
    aggregate_and_plot_results(results)

if __name__ == "__main__":
    main()
