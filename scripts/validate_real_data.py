import os
import mne
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, precision_score, recall_score, f1_score)
import logging
from tqdm import tqdm
import time
import toml
from utils import compute_eeg_spectra, FineTuneResNet, create_spectra_image, binarize

all_accuracies = []
all_aucs = []
all_precisions = []
all_recalls = []
all_f1s = []
all_fprs = []
all_tprs = []
participant_labels = []
survey_counts = []
all_cms = []
all_probs = []

# Load general configuration from the main TOML file
config_fn = 'config.toml'
config = toml.load(config_fn)

np.random.seed(42)  # Set your preferred seed
torch.manual_seed(42)  # For CPU
torch.cuda.manual_seed(42)  # For current GPU
torch.cuda.manual_seed_all(42)  # If you use multiple GPUs

# Extract configurations from the TOML file
output_dir = config['general']['resnet_output_dir']
noise_threshold = config['model']['noise_threshold']
n_permutations = config['model']['n_permutations']
image_size = config['model']['image_size']
participants = config['general']['participants']
vae_RCS0X = config['general']['vae_participant_id']
val_output_dir = config['general']['val_output_dir']
val_data_folder = config['general']['val_data_folder']

# Create the output directory if it does not exist
os.makedirs(val_output_dir, exist_ok=True)

# Setup a single log file for all participants
log_file = f"{val_output_dir}all_ppt_val_log_file.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def classify_spectra_images(image):
    """
    Classify an EEG spectra image using the trained ResNet model.

    Parameters:
    - image (PIL.Image): The EEG spectra image to classify.

    Returns:
    - predicted_class (int): Predicted class label (0 for clean, 1 for noisy).
    - noise_prob (float): Probability of the image being noisy.
    - classification_time (float): Time taken to classify the image.
    """
    start_time = time.time()

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Classify the processed image
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)

    # Get the probability of the noisy class (assuming the second class is "noisy")
    noise_prob = probabilities[0][1].item()

    # Use the custom threshold for classification
    predicted_class = 1 if noise_prob > noise_threshold else 0

    end_time = time.time()  # End timing

    classification_time = end_time - start_time  # Calculate time taken

    return predicted_class, noise_prob, classification_time

def classify_and_label_images(image, output_dir, img_name, noise_threshold=0.2):
    """
    Classify an image and label it with the prediction and probability.

    Parameters:
    - image (PIL.Image): The EEG spectra image to classify.
    - output_dir (str): Directory to save the labeled image.
    - img_name (str): Name for the output image file.
    - noise_threshold (float, optional): Threshold for classifying the image as noisy. Default is 0.2.

    Returns:
    - label (str): Classification label ("NOISY" or "CLEAN").
    - noise_prob (float): Probability of the image being noisy.
    """
    img_tensor = transform(image).unsqueeze(0).to(device)

    preprocessed_img = img_tensor.squeeze(0).cpu().numpy()
    preprocessed_img = (preprocessed_img * 255).astype('uint8')
    preprocessed_img = Image.fromarray(preprocessed_img[0], mode='L')

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)

    noise_prob = probs[0][1].item()
    label = 'NOISY' if noise_prob > noise_threshold else 'CLEAN'
    color = 'red' if noise_prob > noise_threshold else 'black'

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    label_text = f"{label}\n{noise_prob:.12f}"
    draw.text((10, 10), label_text, fill=color, font=font)

    processed_img_path = os.path.join(output_dir, img_name)
    image.save(processed_img_path)

    return label, noise_prob

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(binarize)
])

def validate_model_on_eeg_files(data_folder, output_folder, RCS0X, save_images=False):
    """
    Validate the trained ResNet model on real EEG data.

    Parameters:
    - data_folder (str): Directory containing EEG data files.
    - output_folder (str): Directory to save validation results and figures.
    - RCS0X (str): Participant identifier.
    - save_images (bool, optional): Flag to save labeled images. Default is False.

    Returns:
    - None
    """
    file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.fif') and "ttls" not in f]
    y_true, y_pred, y_probs = [], [], []
    total_time = 0
    num_images = 0
    full_pipe_time = 0

    processed_output_folder = os.path.join(output_folder, 'processed_images/')
    os.makedirs(processed_output_folder, exist_ok=True)

    for file_path in file_paths:
        full_pipe_start = time.time()
        file_name = os.path.basename(file_path)
        true_label = 1 if 'during' in file_name else 0  # Placeholder logic for true labels
        y_true.append(true_label)

        freqs, spectra = compute_eeg_spectra(file_path)
        image = create_spectra_image(freqs, spectra)

        img_name = f"{file_name}_processed.png"
        predicted_label, noise_prob, classification_time = classify_spectra_images(image)
        full_pipe_end = time.time()

        # Update the total time and image count
        total_time += classification_time
        full_pipe_time += full_pipe_end - full_pipe_start
        num_images += 1

        y_pred.append(predicted_label)
        y_probs.append(noise_prob)

        # Save labeled images only if save_images is True
        if save_images:
            classify_and_label_images(image, processed_output_folder, img_name, noise_threshold=noise_threshold)

    accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"Accuracy: {accuracy:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    logging.info(f"ROC AUC: {roc_auc:.4f}")

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Calculate average classification time per image
    average_time = total_time / num_images if num_images > 0 else 0
    logging.info(f"Total classification time for participant {RCS0X}: {total_time:.4f} seconds")
    logging.info(f"Average classification time per image: {average_time:.4f} seconds")

    average_full_pipe_time = full_pipe_time / num_images if num_images > 0 else 0
    logging.info(f"Total full pipeline time for participant {RCS0X}: {full_pipe_time:.4f} seconds")
    logging.info(f"Average full pipeline time per image: {average_full_pipe_time:.4f} seconds")

    # Permutation Testing for AUC p-value

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Clean', 'Noisy']).plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.pdf'))
    plt.close()

    #save y_true and y_pred
    np.save(os.path.join(output_folder, 'y_true.npy'), y_true)
    np.save(os.path.join(output_folder, 'y_pred.npy'), y_pred)

    og_y_true = y_true

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.00])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_folder, 'roc_curve.png'))
    plt.savefig(os.path.join(output_folder, 'roc_curve.pdf'))
    plt.close()

    permuted_aucs = []
    logging.info(f'Starting permutation testing with {n_permutations} permutations')
    for _ in tqdm(range(n_permutations)):
        np.random.shuffle(y_true)
        fpr_perm, tpr_perm, _ = roc_curve(y_true, y_probs)
        permuted_auc = auc(fpr_perm, tpr_perm)
        permuted_aucs.append(permuted_auc)

    permuted_aucs = np.array(permuted_aucs)
    p_value = np.mean(permuted_aucs >= roc_auc)
    logging.info(f'Permutation test p-value for AUC: {p_value:.8f}')

    all_accuracies.append(accuracy)
    all_aucs.append(roc_auc)
    all_f1s.append(f1)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_fprs.append(fpr)
    all_tprs.append(tpr)
    participant_labels.append(RCS0X)
    survey_counts.append(len(y_true))
    all_cms.append(cm)
    all_probs.append(y_probs)

    with open(os.path.join(output_folder, "classification_report.txt"), "w") as f:
        f.write("Classification Report\n")
        f.write(f"\nAccuracy: {accuracy:.4f}")
        f.write(f"\nROC AUC: {roc_auc:.4f}")
        f.write(f"\nPermutation test p-value for AUC: {p_value:.4f}")

    plt.figure()
    plt.hist(permuted_aucs, bins=30, color='gray', alpha=0.7, label='Permuted AUCs')
    plt.axvline(roc_auc, color='red', linestyle='dashed', linewidth=2, label='Actual AUC')
    plt.xlabel('AUC')
    plt.ylabel('Frequency')
    plt.title('Permutation Test for AUC')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'permutation_test_auc.png'))
    plt.savefig(os.path.join(output_folder, 'permutation_test_auc.pdf'))
    plt.close()

# Run validation for all participants
for RCS0X in participants:
    logging.info(f"STARTING WITH PARTICIPANT: {RCS0X}")

    data_folder = val_data_folder.format(RCS0X)
    output_folder = f"{val_output_dir}pre_during_performance_figs/{RCS0X}/"

    os.makedirs(output_folder, exist_ok=True)

    # Define and load the ResNet model before calling the validation function
    model = FineTuneResNet()
    model_path = os.path.join(output_dir, f"{vae_RCS0X}_final_resnet_model.pth")  # Dynamically set the model path

    # Load the model state dict and remove 'module.' prefix if it exists (from DataParallel)
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)  # Load the trained model
    model = model.to(device)
    model.eval()  # Ensure the model is in evaluation mode

    # Validate model on EEG files
    validate_model_on_eeg_files(data_folder, output_folder, RCS0X, save_images=True)

# Summary statistics across participants
logging.info("SUMMARY STATISTICS ACROSS PARTICIPANTS")
logging.info(f"Min Accuracy: {min(all_accuracies):.4f}, Max Accuracy: {max(all_accuracies):.4f}")
logging.info(f"Min AUC: {min(all_aucs):.4f}, Max AUC: {max(all_aucs):.4f}")
logging.info(f"Min Precision: {min(all_precisions):.4f}, Max Precision: {max(all_precisions):.4f}")
logging.info(f"Min Recall: {min(all_recalls):.4f}, Max Recall: {max(all_recalls):.4f}")

output_dir = f"{val_output_dir}all_participants_performance/"
os.makedirs(output_dir, exist_ok=True)


np.save(f"{output_dir}all_accuracies.npy", all_accuracies)
np.save(f"{output_dir}all_aucs.npy", all_aucs)
np.save(f"{output_dir}all_precisions.npy", all_precisions)
np.save(f"{output_dir}all_recalls.npy", all_recalls)

#convert fprs and tprs into dict
fpr_dict = {}
tpr_dict = {}
probs_dict = {}
for i, ppt in enumerate(participant_labels):
    fpr_dict[ppt] = all_fprs[i]
    tpr_dict[ppt] = all_tprs[i]
    probs_dict[ppt] = all_probs[i]

np.save(f"{output_dir}fpr_dict.npy", fpr_dict)
np.save(f"{output_dir}tpr_dict.npy", tpr_dict)
np.save(f"{output_dir}probs_dict.npy", probs_dict)
np.save(f"{output_dir}participant_labels.npy", participant_labels)
np.save(f"{output_dir}survey_counts.npy", survey_counts)
np.save(f"{output_dir}all_cms.npy", all_cms)
