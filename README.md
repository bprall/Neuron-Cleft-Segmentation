# CREMI-Neuron-Cleft-Segmentation
Synaptic Cleft Segmentation

All of the code needed to train, create, and display a synaptic cleft segmentation mask are available in the notebook titled: "main.ipynb". 

# Data
In order to train the model, generate a mask, or diplay the pre-generated mask, you must first download a 3D EM volume, and you can download the data I used [here](https://cremi.org/data/). The model was trained on the "cropped version" of "Dataset A."

# Package Dependencies
```
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

# Commands
To install dependencies
```
pip3 install -r requirements.txt
```

To run the application
```
python app.py <path_to_waveform> <path_to_model_dict>
```

To test the model
```
python test.py <path_to_model_dict>
```

To train the model
```
python train.py
```

# Components
The code cosists of multiple components, some of the more significant for training and segmentation purposes include: histogram equalization, gaussian thresholding, a min/max normalization function, a custom function to extract subvolumes of a desired size, a custom dataloader with a transform arg that is currently set to take custom class "Transformations" which randomly flips and rotates subvolumes and labels (this can be disabled by setting "transform=None"), a U-Net3D-like model that takes input arg n_filters to determine how many filters the model should use, a 3D IoU loss function, a 3D Dice loss function, the pytorch BCELoss function, a custom training and validation cell that displays real-time training-loss and validation-loss data alongside a progress tracker for each epoch, a custom voxel-wise testing cell, a custom mask generation cell that uses "sliding window" across the input data to generate a single mask, and a cell for displaying the generated mask onto the raw input data. 


A pre-trained model dict and pre-generated mask can be found [here](https://drive.google.com/drive/folders/1ML912JIxkp9qZ_mLMdHyseDSaqw0EPec?usp=share_link). The file, "final_trained_synapse_model.pth", is a trained model state dictionary that can be used for mask generation. The file "segmentation_mask.hdf" is a pre-generated segmentation mask that can be directly loaded into the display cell. Simply save these files to the same directory as the notebook, and they are already set up in the notebook to be uploaded.

All of the hyperparameters for training and mask generation are held in the second cell of the notebook with annotations detailing what they do. It is essential for the functon of mask generation and testing that these parameters are the same as they were for model training. The current values are equal to that of which "final_trained_synapse_model.pth" was trained on and "segmentation_mask.hdf" was generated on.

To run mask generation or display a generated mask using a the pre-trained model or pre-generated mask, you must first run cells 1-11 to load and pre-process the input data, you can then run cell 25 to load the pre-generated segmentaion mask for display. If you want to generate your own mask, run cell 19 so that the model can be instantiated, then you can go down to and run cell 24 to generate your mask (this will overwrite the "segmentation_mask.hdf" file if you have it in the directory). After "Finished Mask Generation" is printed, the mask can be displayed alongside the raw data by running cell 25. This final cell will instantiate a local server running neuroglancer for displaying the volumetric data (you will need to set the "PORT" parameter to an unused port number; it is 9000 by default).

# Contributions

All code was written by Blake Prall.
