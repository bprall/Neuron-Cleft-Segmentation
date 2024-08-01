import torch
import h5py
import numpy as np
import cv2
from torch.utils.data import Dataset
import random

# Use cuda computation if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('Cache cleared')
print(device)

# Function to normalize data
def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    array = (array - min_val) / (max_val - min_val)
    return array

# Data loading and preprocessing
def load_and_preprocess_data(file_path):
    with h5py.File(file_path, 'r') as f:
        volume = f['volumes/raw'][:]
        labels = f['volumes/labels/clefts'][:]

    labels = labels.astype(np.float32)
    volume = volume.astype(np.uint8)

    labels[labels < 10000] = 1
    labels[labels > 10000] = 0

    equilized_volume = np.zeros(volume.shape)
    for z in range(volume.shape[0]):
        equilized_volume[z] = cv2.equalizeHist(volume[z].astype(np.uint8))

    thresholded_volume = np.zeros(equilized_volume.shape)
    kernel_size = 21
    for z in range(equilized_volume.shape[0]):
        volume_blur = cv2.GaussianBlur(equilized_volume[z], (kernel_size, kernel_size), kernel_size)
        volume_blur = equilized_volume[z] / volume_blur
        thresholded_volume[z] = (((volume_blur - volume_blur.min()) / float(volume_blur.max() - volume_blur.min())) * 255).astype(np.uint8)

    equilized_volume = np.zeros(thresholded_volume.shape)
    for z in range(thresholded_volume.shape[0]):
        equilized_volume[z] = cv2.equalizeHist(thresholded_volume[z].astype(np.uint8))

    volume = normalize(equilized_volume)
    return volume, labels

# Extract subvolumes
def extract_subvolumes(volume, label_volume, subvol_shape):
    subvolumes = []
    label_subvolumes = []
    for z in range(0, volume.shape[0], subvol_shape[0]):
        for y in range(0, volume.shape[1], subvol_shape[1]):
            for x in range(0, volume.shape[2], subvol_shape[2]):
                subvol = volume[z:z + subvol_shape[0], y:y + subvol_shape[1], x:x + subvol_shape[2]]
                label_subvol = label_volume[z:z + subvol_shape[0], y:y + subvol_shape[1], x:x + subvol_shape[2]]
                if subvol.shape == tuple(subvol_shape):
                    subvolumes.append(subvol)
                    label_subvolumes.append(label_subvol)
    return subvolumes, label_subvolumes

# Dataset class
class SubvolumeDataset(Dataset):
    def __init__(self, subvolumes, labels, transform=None):
        self.subvolumes = subvolumes
        self.labels = labels
        self.transform = transform
        if self.transform is not None:
            self.subvolumes = subvolumes * 5
            self.labels = labels * 5

    def __len__(self):
        return len(self.subvolumes)

    def __getitem__(self, idx):
        subvol_tensor = self.subvolumes[idx]
        label_subvol_tensor = self.labels[idx]
        subvol_tensor = torch.from_numpy(subvol_tensor)
        label_subvol_tensor = torch.from_numpy(label_subvol_tensor)
        if self.transform:
            transformed_labels, transformed_volume = self.transform(label_subvol_tensor, subvol_tensor)
            return transformed_volume, transformed_labels
        else:
            return subvol_tensor, label_subvol_tensor

# Transformations class
class Transformations(object):
    def __call__(self, labels, volume):
        transformed_labels, transformed_volume = self.apply_transformations(labels, volume)
        return transformed_labels, transformed_volume

    def apply_transformations(self, labels, volume):
        flip_lr = random.random() < 0.5
        flip_ud = random.random() < 0.5
        flip_fb = random.random() < 0.5

        if flip_lr:
            labels = torch.flip(labels, dims=(2,))
            volume = torch.flip(volume, dims=(2,))
        if flip_ud:
            labels = torch.flip(labels, dims=(1,))
            volume = torch.flip(volume, dims=(1,))
        if flip_fb:
            labels = torch.flip(labels, dims=(0,))
            volume = torch.flip(volume, dims=(0,))

        num_rotations = random.randint(0, 3)
        for _ in range(num_rotations):
            labels = torch.rot90(labels, k=1, dims=(1, 2))
            volume = torch.rot90(volume, k=1, dims=(1, 2))

        return labels, volume
