import torch.nn as nn
import mne
import torch
from torchvision import models
from PIL import Image, ImageDraw, ImageFont
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model for generating synthetic EEG data.

    Attributes:
    - encoder (nn.Sequential): Encoder network.
    - fc_mu (nn.Linear): Linear layer to compute the mean of the latent space.
    - fc_logvar (nn.Linear): Linear layer to compute the log variance of the latent space.
    - decoder (nn.Sequential): Decoder network.
    """
    def __init__(self, data_len=2500, kernel_len=15, pad_len=3, stride_len=1, latent_factors=50):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=kernel_len, stride=stride_len, padding=pad_len),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=kernel_len, stride=stride_len, padding=pad_len),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=kernel_len, stride=stride_len, padding=pad_len),
            nn.ReLU(),
            nn.Flatten()
        )
        output_size = self.calculate_output_size(data_len, kernel_len, pad_len, stride_len, layers=3)
        self.fc_mu = nn.Linear(512 * output_size, latent_factors)
        self.fc_logvar = nn.Linear(512 * output_size, latent_factors)
        self.decoder = nn.Sequential(
            nn.Linear(latent_factors, 512 * output_size),
            nn.ReLU(),
            nn.Unflatten(1, (512, output_size)),
            nn.ConvTranspose1d(512, 256, kernel_size=kernel_len, stride=stride_len, padding=pad_len),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=kernel_len, stride=stride_len, padding=pad_len),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 1, kernel_size=kernel_len, stride=stride_len, padding=pad_len)
        )

    def calculate_output_size(self, input_size, kernel_len, pad_len, stride_len, layers):
        """
        Calculate the output size after convolutional layers.

        Parameters:
        - input_size (int): Size of the input data.
        - kernel_len (int): Kernel size of the convolutional layers.
        - pad_len (int): Padding length.
        - stride_len (int): Stride length.
        - layers (int): Number of convolutional layers.

        Returns:
        - int: Output size after the convolutional layers.
        """
        size = input_size
        for _ in range(layers):
            size = (size + 2 * pad_len - kernel_len) // stride_len + 1
        return size

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent space.

        Parameters:
        - mu (torch.Tensor): Mean of the latent space.
        - logvar (torch.Tensor): Log variance of the latent space.

        Returns:
        - torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE.

        Parameters:
        - x (torch.Tensor): Input data.

        Returns:
        - x_recon (torch.Tensor): Reconstructed input.
        - mu (torch.Tensor): Mean of the latent space.
        - logvar (torch.Tensor): Log variance of the latent space.
        """
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


# Define the ResNet model
class FineTuneResNet(nn.Module):
    """
    Fine-tuned ResNet-18 model for binary classification of EEG spectra images.

    Attributes:
    - resnet (torchvision.models.ResNet): Modified ResNet-18 model.
    """
    def __init__(self):
        super(FineTuneResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # Adjust the final layer for binary classification

    def forward(self, x):
        """
        Forward pass through the ResNet model.

        Parameters:
        - x (torch.Tensor): Input image tensor.

        Returns:
        - torch.Tensor: Output logits from the model.
        """
        return self.resnet(x)

def binarize(img, threshold=0.8):
    """
    Binarize an image based on a given threshold.

    Parameters:
    - img (torch.Tensor): Input image tensor.
    - threshold (float, optional): Threshold value. Default is 0.8.

    Returns:
    - torch.Tensor: Binarized image tensor.
    """
    return (img > threshold).float()


def compute_eeg_spectra(file_path, sfreq=1000):
    """
    Compute the power spectral density (PSD) of EEG data from a file.

    Parameters:
    - file_path (str): Path to the EEG data file.
    - sfreq (int, optional): Sampling frequency. Default is 1000 Hz.

    Returns:
    - tuple:
        - freqs (numpy.ndarray): Array of frequency bins.
        - Pxx (numpy.ndarray): PSD values for each frequency bin.
    """
    raw = mne.io.read_raw_fif(file_path, verbose=False)
    all_data = raw.get_data()
    current_window = all_data[:, :int(sfreq * 2.5)]
    f, Pxx = signal.welch(current_window, fs=sfreq, nperseg=1024)
    return f, Pxx

def create_spectra_image(freqs, spectra):
    """
    Create an image of the EEG spectra.

    Parameters:
    - freqs (numpy.ndarray): Array of frequency bins.
    - spectra (numpy.ndarray): PSD values for each channel.

    Returns:
    - PIL.Image: Image of the EEG spectra.
    """
    plt.switch_backend('Agg')
    fig, ax = plt.subplots()
    for channel in range(spectra.shape[0]):
        ax.plot(freqs, np.log10(spectra[channel]), color='black', alpha=0.5, lw=0.5)
    ax.set_xlim([0, 150])
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    #also save pdf version
    plt.savefig(buf, format='pdf', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf).convert('L')

# EarlyStopping class
class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting during training.

    Attributes:
    - patience (int): Number of epochs to wait before stopping.
    - min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    - counter (int): Counter for epochs without improvement.
    - best_loss (float): Best validation loss observed.
    - early_stop (bool): Flag indicating whether early stopping has been triggered.
    """
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, val_loss):
        """
        Check if training should be stopped early based on validation loss.

        Parameters:
        - val_loss (float): Current validation loss.

        Returns:
        - bool: True if training should stop, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

