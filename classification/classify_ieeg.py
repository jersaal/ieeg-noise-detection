import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import toml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import FineTuneResNet, compute_eeg_spectra, create_spectra_image, binarize

# Load configuration from TOML file
config_fn = 'classify_config.toml'
config = toml.load(config_fn)

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Extract configuration settings
noise_threshold = config['model']['noise_threshold']
output_folder = config['paths']['output_folder']
data_folder = config['paths']['data_folder']
model_path = config['paths']['model_path']

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

image_size = 672

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load pre-trained ResNet model
model = FineTuneResNet()

# Load the model state dict and handle 'module.' prefix (if DataParallel was used)
state_dict = torch.load(model_path)
new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(binarize)
])

def classify_spectra_images(image):
    """
    Classify an EEG spectra image using the trained ResNet model.
    Returns predicted_class (0: clean, 1: noisy) and noise_prob.
    """
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        noise_prob = probabilities[0][1].item()
        predicted_class = 1 if noise_prob > noise_threshold else 0

    return predicted_class, noise_prob

def classify_eeg_files(data_folder, output_folder):
    """
    Classify iEEG files using the trained ResNet model.
    Saves classification results (filename, noise classification, noise probability) in a CSV.
    """
    results = []
    file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.fif')]

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")

        freqs, spectra = compute_eeg_spectra(file_path)
        image = create_spectra_image(freqs, spectra)

        predicted_label, noise_prob = classify_spectra_images(image)

        # Append results for CSV
        results.append({
            'filename': file_name,
            'noise': bool(predicted_label),
            'noise_prob': noise_prob
        })

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    csv_save_path = os.path.join(output_folder, 'classification_results.csv')
    results_df.to_csv(csv_save_path, index=False)
    print(f"Results saved to {csv_save_path}")

# Run the validation
classify_eeg_files(data_folder, output_folder)
