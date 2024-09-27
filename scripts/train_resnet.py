import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import logging
import toml
from tqdm import tqdm
from utils import EarlyStopping, FineTuneResNet, binarize

# Load general configuration from the main TOML file
config_fn = 'config.toml'
config = toml.load(config_fn)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Extract configurations from the TOML file
RCS0X = config['general']['vae_participant_id']
image_size = config['resnet']['image_size']
base_dir = config['general']['base_dir']
output_dir = config['general']['resnet_output_dir']
learning_rate = config['resnet']['learning_rate']
batch_size = config['resnet']['batch_size']
num_epochs = config['resnet']['num_epochs']
noise_threshold = config['model']['noise_threshold']

# Setup directories
synthetic_data_dir = os.path.join(base_dir, 'spectra/spectra_figures/')
os.makedirs(synthetic_data_dir, exist_ok=True)
fig_fold = os.path.join(base_dir, 'large_resnet_model_figs/')
os.makedirs(fig_fold, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Setup logging
logging_fn = os.path.join(output_dir, f"{RCS0X}_resnet_training_log.txt")
logging.basicConfig(filename=logging_fn, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
logging.info(f'Using device: {device}')
logging.info(f'Number of GPUs available: {num_gpus}')

# Define transforms for the image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(binarize)
])

# Custom Dataset for loading and saving images
class CustomImageDataset(Dataset):
    """
    Custom Dataset for loading and preprocessing EEG spectra images.

    Attributes:
    - img_dir (str): Directory containing the image data.
    - transform (callable, optional): Transformation to apply to the images.
    - img_files (list of str): List of image file paths.
    - img_labels (list of int): List of labels corresponding to the images.
    - save_preprocessed (bool): Flag to save preprocessed images.
    - save_dir (str, optional): Directory to save preprocessed images.
    """
    def __init__(self, img_dir, transform=None, save_preprocessed=False, save_dir=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = []
        self.img_labels = []
        self.save_preprocessed = save_preprocessed
        self.save_dir = save_dir

        for label_dir in os.listdir(img_dir):
            label_path = os.path.join(img_dir, label_dir)
            if os.path.isdir(label_path):
                label = 1 if 'noisy' in label_dir else 0
                for img_file in os.listdir(label_path):
                    if os.path.isfile(os.path.join(label_path, img_file)):
                        self.img_files.append(os.path.join(label_path, img_file))
                        self.img_labels.append(label)

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.img_files)

    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset at the given index.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - image (torch.Tensor): Transformed image tensor.
        - label (int): Label of the image (0 for clean, 1 for noisy).
        """
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("L")
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)

        # Save pre-processed images as PDFs if the flag is set
        if self.save_preprocessed and self.save_dir:
            save_path = os.path.join(self.save_dir, f"{os.path.basename(img_path).split('.')[0]}_preprocessed.pdf")
            plt.figure()
            plt.imshow(image[0].numpy(), cmap='gray')  # Image is now a Tensor, extract the first channel
            plt.axis('off')
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.close()

        return image, label

# Function for K-fold cross-validation training
def train_k_fold(model_class, dataset, batch_size, criterion, optimizer_class, num_epochs, k_folds=5):
    """
    Perform K-fold cross-validation training.

    Parameters:
    - model_class (nn.Module): Class of the model to instantiate.
    - dataset (Dataset): Dataset to perform cross-validation on.
    - batch_size (int): Batch size for training.
    - criterion (callable): Loss function.
    - optimizer_class (torch.optim.Optimizer): Optimizer class.
    - num_epochs (int): Number of epochs for training.
    - k_folds (int, optional): Number of folds for cross-validation. Default is 5.

    Returns:
    - None
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_labels, all_preds = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Training fold {fold+1}/{k_folds}")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        # Initialize model
        model = model_class().to(device)
        if num_gpus > 1:
            model = nn.DataParallel(model)
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)

        # Train and validate model
        print("going to start train and validate")
        fold_labels, fold_preds= train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs)

        all_labels.extend(fold_labels)
        all_preds.extend(fold_preds)

    # Plot ROC curve and confusion matrix
    plot_roc(all_labels, all_preds)
    plot_confusion_matrix(all_labels, all_preds)

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Train the model and validate it on the validation set.

    Parameters:
    - model (nn.Module): Model instance to train.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - criterion (callable): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - num_epochs (int): Number of epochs for training.

    Returns:
    - all_labels (list): True labels from the validation set.
    - all_preds (list): Predicted probabilities for the positive class.
    """
    early_stopping = EarlyStopping(patience=5)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Early stopping check on validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()  # Accumulate validation loss

        # Calculate average validation loss
        val_loss /= len(val_loader)

        # Check for early stopping
        if early_stopping.step(val_loss):
            print("Early stopping triggered.")
            break

    # After training completes (either early stop or max epochs), evaluate the final model on validation set
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = nn.functional.softmax(outputs, dim=1)  # Assuming binary classification
            all_labels.extend(labels.cpu().numpy())  # Store true labels
            all_preds.extend(probs[:, 1].cpu().numpy())  # Store predicted probabilities for class 1

    # Return the final labels, predictions, AUC, and ROC curve
    return all_labels, all_preds


# Function for training on the entire dataset
def train_on_all_data(model_class, dataset, batch_size, criterion, optimizer_class, num_epochs):
    """
    Train the model on the entire dataset.

    Parameters:
    - model_class (nn.Module): Class of the model to instantiate.
    - dataset (Dataset): Dataset to train on.
    - batch_size (int): Batch size for training.
    - criterion (callable): Loss function.
    - optimizer_class (torch.optim.Optimizer): Optimizer class.
    - num_epochs (int): Number of epochs for training.

    Returns:
    - None
    """
    model = model_class().to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save the trained model
    model_save_path = os.path.join(output_dir, f"{RCS0X}_final_resnet_model.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved at {model_save_path}")

# Function to plot ROC curve
def plot_roc(all_labels, all_preds):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters:
    - all_labels (list): True labels.
    - all_preds (list): Predicted probabilities for the positive class.

    Returns:
    - None
    """
    fpr, tpr, _ = roc_curve(all_labels, all_preds, pos_label=1)
    roc_auc = auc(fpr, tpr)
    logging.info(f"AUC: {roc_auc}")
    logging.info(f"False Positive Rate: {fpr}")
    logging.info(f"True Positive Rate: {tpr}")
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(fig_fold, 'roc_curve.png'))
    plt.savefig(os.path.join(fig_fold, 'roc_curve.pdf'))
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(all_labels, all_preds):
    """
    Plot the confusion matrix.

    Parameters:
    - all_labels (list): True labels.
    - all_preds (list): Predicted probabilities for the positive class.

    Returns:
    - None
    """
    logging.info('Plotting confusion matrix')
    y_pred = np.array([1 if p > 0.5 else 0 for p in all_preds])
    cm = confusion_matrix(all_labels, y_pred)
    logging.info(f"Confusion Matrix: {cm}")
    logging.info(f"Accuracy: {(cm[0, 0] + cm[1, 1]) / np.sum(cm)}")
    logging.info(f"Precision: {cm[1, 1] / (cm[1, 1] + cm[0, 1])}")
    logging.info(f"Recall: {cm[1, 1] / (cm[1, 1] + cm[1, 0])}")
    logging.info(f"F1 Score: {2 * cm[1, 1] / (2 * cm[1, 1] + cm[0, 1] + cm[1, 0])}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Clean', 'Noisy'])
    disp.plot(cmap=plt.cm.Blues)
    plt.xlim
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(fig_fold, 'confusion_matrix.png'))
    plt.savefig(os.path.join(fig_fold, 'confusion_matrix.pdf'))
    plt.close()

# Main function
def main():
    """
    Main function to train the ResNet classifier.

    Performs the following steps:
    - Loads the synthetic dataset.
    - Sets up the model, criterion, and optimizer.
    - Performs K-fold cross-validation.
    - Trains on the entire dataset.
    - Saves the trained model and logs.

    Returns:
    - None
    """
    spec_figfold = os.path.join(output_dir, 'synthetic_spectra_figures/')
    os.makedirs(spec_figfold, exist_ok=True)
    synthetic_dataset = CustomImageDataset(img_dir=synthetic_data_dir, transform=transform, \
            save_preprocessed=False, save_dir=spec_figfold)
    model_class = FineTuneResNet
    criterion = nn.CrossEntropyLoss()
    optimizer_class = optim.Adam

    # Separate functions for K-fold cross-validation and full dataset training
    run_kfold = True  # Set this to False to skip k-fold validation
    run_full_data = True  # Set this to False to skip full data training

    if run_kfold:
        logging.info('Starting K-fold cross-validation')
        train_k_fold(model_class, synthetic_dataset, batch_size, criterion, optimizer_class, num_epochs)

    if run_full_data:
        logging.info('Training on the entire dataset')
        train_on_all_data(model_class, synthetic_dataset, batch_size, criterion, optimizer_class, num_epochs)

if __name__ == "__main__":
    main()
