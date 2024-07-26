import argparse
import torch
from model import SynapseSegmentationModel
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.constants import (
    NUM_FILTERS, BATCH_SIZE, SAMPLE_SHAPE, THRESHOLD
)
from utils.data_preprocessing import (
    device, load_and_preprocess_data, extract_subvolumes, SubvolumeDataset, Transformations
)

def main(data_path, model_path):
    # Load and preprocess data
    volume, labels = load_and_preprocess_data(data_path)
    volume_subvols, label_subvols = extract_subvolumes(volume, labels, SAMPLE_SHAPE)

    # Split data
    train_volume, val_test_volume, train_label, val_test_label = train_test_split(volume_subvols, label_subvols, test_size=0.25, random_state=123)
    val_volume, test_volume, val_label, test_label = train_test_split(val_test_volume, val_test_label, test_size=0.5, random_state=123)

    # Data loaders
    transform = Transformations()
    test_loader = DataLoader(SubvolumeDataset(test_volume, test_label, transform=transform), batch_size=BATCH_SIZE, shuffle=True)

    # Define a function to compute the accuracy
    def compute_accuracy(predictions, labels):
        # Flatten predictions and labels
        predictions = predictions.view(-1)
        labels = labels.view(-1)
        # Compute the number of correct predictions
        num_correct = (predictions == labels).sum().item()
        # Compute the total number of predictions
        num_total = labels.numel()
        # Compute the accuracy
        accuracy = num_correct / num_total
        return accuracy

    # Initialize the model
    model = SynapseSegmentationModel(n_filters=NUM_FILTERS)

    # Load the saved state dictionary
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    total_accuracy = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device).unsqueeze(1)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs.float())

            # Convert the outputs to class predictions
            outputs = (outputs >= THRESHOLD).float()  # Convert to binary

            # Compute the accuracy for this batch
            batch_accuracy = compute_accuracy(outputs, targets)

            # Add batch accuracy to total accuracy
            total_accuracy += batch_accuracy * inputs.size(0)
            total_samples += inputs.size(0)

    # Divide by the total number of examples to get the overall accuracy
    overall_accuracy = total_accuracy / total_samples

    print(f"Test accuracy: {overall_accuracy:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained segmentation model on a test set.')
    parser.add_argument('data_path', type=str, help='Path to the input data file')
    parser.add_argument('model_path', type=str, help='Path to the saved model state dictionary')
    args = parser.parse_args()
    
    main(args.data_path, args.model_path)
