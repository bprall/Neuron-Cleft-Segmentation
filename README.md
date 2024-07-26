# CREMI-Neuron-Cleft-Segmentation

**Synaptic Cleft Segmentation via a Custom CNN**

## Usage

All the code needed to train, create, and display a synaptic cleft segmentation mask is available in the notebook titled "main.ipynb." Alternatively, you can use the provided scripts, which are described below.

**Note:** You will need to separately download the dataset and optionally the trained model weights and pre-generated mask to run the program.

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

# Components

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

To use these files with the notebook, run cells 1-11 to load and pre-process the input data, then run cell 25 to display the pre-generated mask. To generate your own mask, run cell 19 to instantiate the model and cell 24 to generate the mask (this will overwrite segmentation_mask.hdf if it is in the directory). After "Finished Mask Generation" is printed, display the mask alongside the raw data by running cell 25. This final cell starts a local server using Neuroglancer to display the volumetric data (set the PORT parameter to an unused port number; default is 9000).

# Commands
To install dependencies
```
pip3 install -r requirements.txt
```

To run the application
```
python app.py <path_to_data> <path_to_model_dict>
```

To test the model
```
python test/test.py <path_to_data> <path_to_model_dict>
```

To train the model
```
python train/train.py <path_to_data>
```

# Contributions

All code was written by Blake Prall.