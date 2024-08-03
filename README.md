# CREMI-Neuron-Cleft-Segmentation

**Synaptic Cleft Segmentation via a Custom CNN**

## Overview

**Report**: [Mapping Neural Circuits: Synaptic Segmentation](report.pdf)  
**Author**: Blake Prall

### Summary
This project explores the use of deep-learning neural networks to map the human connectome, focusing on the segmentation of neuronal synaptic clefts. The central aim is to develop a neural network architecture capable of synaptic cleft segmentation. The hypothesis is that a U-Net3D-like architecture will effectively achieve this task.

The full report is available in `report.pdf` and provides an in-depth look at the project's development and outcomes. Additionally, you can download the development notebook, notebook.ipynb, to gain deeper insights into the program’s functionality and implementation.


## Usage

All the code needed to train, create, and display a synaptic cleft segmentation mask is available in the scripts and commands described below.  Additionally, you can download the [development notebook](), `notebook.ipynb`, to gain deeper insights into the program’s functionality and implementation.

**Note:** You will need to separately download the [dataset](https://cremi.org/data/) and optionally the [trained model weights and pre-generated mask](https://drive.google.com/drive/folders/1ML912JIxkp9qZ_mLMdHyseDSaqw0EPec?usp=share_link) to run the program. Also, to use the pre-trained model, CUDA must be set to GPU. Ensure that your environment is properly configured with GPU support to utilize the pre-trained model effectively.

## Data

To train the model, generate a mask, or display a pre-generated mask, download a 3D EM volume from [CREMI](https://cremi.org/data/) and place it in the root. The model was trained on the "cropped version" of "Dataset A."

## Package Dependencies

```plaintext
CV2
h5py
Numpy
Random
PyTorch
TorchVision
SK-Learn
Neuroglancer
MatPlotLib
```

## Components

The code includes several components for training and segmentation:

- **Histogram Equalization:** Enhances contrast in images.
- **Gaussian Thresholding:** Applies a Gaussian filter for thresholding.
- **Normalization Function:** Min/max normalization of data.
- **Subvolume Extraction:** Custom function for extracting subvolumes of desired size.
- **Custom DataLoader:** Includes a transform argument for random flipping and rotation (can be disabled by setting transform=None).
- **U-Net3D-like Model:** A 3D U-Net model with a configurable number of filters.
- **3D IoU Loss:** Intersection over Union loss function for 3D data.
- **3D Dice Loss:** Dice loss function for 3D data.
- **BCELoss:** Binary Cross-Entropy Loss function.
- **Training and Validation Cells:** Displays real-time training and validation loss data with a progress tracker.
- **Voxel-wise Testing:** Custom testing function for voxel-wise evaluation.
- **Mask Generation:** Uses a sliding window approach to generate segmentation masks.
- **Mask Display:** Displays the generated mask on raw input data.

A pre-trained model and pre-generated mask can be found [here](https://drive.google.com/drive/folders/1ML912JIxkp9qZ_mLMdHyseDSaqw0EPec?usp=share_link). Save these files to the root:

- `final_trained_synapse_model.pth`: Trained model state dictionary for mask generation.
- `segmentation_mask.hdf`: Pre-generated segmentation mask for display.

## Commands
To install dependencies
```
pip3 install -r requirements.txt
```

To display a pre-generated mask
```
python display.py <path_to_data> <path_to_mask>
```

To generate and display a mask
```
python generate.py <path_to_data> <path_to_model_dict>
```

To test the model
```
python test/test.py <path_to_data> <path_to_model_dict>
```

To train the model
```
python train/train.py <path_to_data>
```

## Notebook

To use the notebook instead of scripts, run cells 1-11 to load and pre-process the input data, then run cell 25 to display the pre-generated mask. To generate your own mask, run cell 19 to instantiate the model and cell 24 to generate the mask (this will overwrite segmentation_mask.hdf if it is in the directory). After "Finished Mask Generation" is printed, display the mask alongside the raw data by running cell 25. This final cell starts a local server using Neuroglancer to display the volumetric data (set the PORT parameter to an unused port number; default is 9000).

## Contributions

All code was written by Blake Prall.
