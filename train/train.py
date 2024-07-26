import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils.constants import (
    NUM_FILTERS, BATCH_SIZE, SAMPLE_SHAPE, NUM_EPOCH, THRESHOLD, BCE_WEIGHT, IOU_WEIGHT, DICE_WEIGHT, CLIP_VALUE, lr
)
from model import SynapseSegmentationModel
from utils.data_preprocessing import (
    device, load_and_preprocess_data, extract_subvolumes, SubvolumeDataset, Transformations
)

# Load and preprocess data
file_path = '../sample_A_20160501.hdf'
volume, labels = load_and_preprocess_data(file_path)
volume_subvols, label_subvols = extract_subvolumes(volume, labels, SAMPLE_SHAPE)

# Split data
train_volume, val_test_volume, train_label, val_test_label = train_test_split(volume_subvols, label_subvols, test_size=0.25, random_state=123)
val_volume, test_volume, val_label, test_label = train_test_split(val_test_volume, val_test_label, test_size=0.5, random_state=123)

# Data loaders
transform = Transformations()
train_loader = DataLoader(SubvolumeDataset(train_volume, train_label, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SubvolumeDataset(val_volume, val_label, transform=transform), batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model
model = SynapseSegmentationModel(n_filters=NUM_FILTERS)

# Move model to the device (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss Functions
def iou_3d_loss(pred, target, threshold=THRESHOLD):
    pred = (pred > threshold).float()
    intersection = torch.logical_and(pred, target).sum(dim=[-3, -2, -1])
    union = torch.logical_or(pred, target).sum(dim=[-3, -2, -1])
    epsilon = 1e-5
    iou = (intersection + epsilon) / (union + epsilon)
    loss = 1.0 - iou.mean()
    return loss

def dice_3d_loss(pred, target, threshold=THRESHOLD):
    pred = (pred > threshold).float()
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)
    intersection = torch.sum(pred * target, dim=1)
    union = torch.sum(pred, dim=1) + torch.sum(target, dim=1)
    epsilon = 1e-5
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    loss = 1.0 - dice
    return loss

criterion = nn.BCELoss()

# Optimization function
optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCH):
    # Train the model for one epoch
    model.train()
    train_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device).unsqueeze(1), targets.to(device).unsqueeze(1)
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs.float())
        
        # Compute the loss
        Dice_loss = dice_3d_loss(outputs, targets).mean()
        IoU_loss = iou_3d_loss(outputs, targets).mean()
        BCE_loss = criterion(outputs, targets).mean()
        loss = (BCE_loss * BCE_WEIGHT) + (IoU_loss * IOU_WEIGHT) + (Dice_loss * DICE_WEIGHT)
        
        # Backward pass, gradient clipping, and optimization step
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_VALUE)
        optimizer.step()
        
        # Accumulate train loss
        train_loss += loss.item()

        # Print status
        if i % 10 == 9 or i+1 == len(train_loader):
            print(f'Train Epoch: {epoch+1} [{(i+1) * len(inputs)}/{len(train_loader.dataset)} ({100. * (i+1) * len(inputs) / len(train_loader.dataset):.0f}%)]\t'
                  f'BCE Loss: {BCE_loss.item():.4f}\tDice Loss: {Dice_loss.item():.6f}\tLoss: {loss.item():.4f}')
            
    # Compute validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device).unsqueeze(1), targets.to(device).unsqueeze(1)
            outputs = model(inputs.float())
            IoU_loss = iou_3d_loss(outputs, targets).mean() * IOU_WEIGHT
            Dice_loss = dice_3d_loss(outputs, targets).mean() * DICE_WEIGHT
            BCE_loss = criterion(outputs, targets).mean() * BCE_WEIGHT
            loss = IoU_loss + Dice_loss + BCE_loss
            val_loss += loss.item()

    # Average train and validation loss
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    # Store train and validation losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Print epoch summary
    print(f'\nEpoch: {epoch+1} Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}\n')
            
# Save final trained model
with open("final_trained_synapse_model.pth", "wb") as f:
    torch.save(model.state_dict(), f)

print("Finished Training")
