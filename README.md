# CREMI-Neuron-Cleft-Segmentation
Synaptic Cleft Segmentation


All of the code needed to train, create, and display a synaptic cleft segmentation mask are available in the notebook titled: "Synaptic Segmentation." In order to train the model, you must first download a 3D EM volume to train on, you can download the data I used [here](https://cremi.org/data/) to either train, or create a cleft mask. The model was trained on the "cropped version" of "Dataset A."

In order to use this notebook, you will need the following libraries: cv2, h5py, numpy, random, torch, torchvision, sklearn, neuroglancer, and matplotlib.

The notebook cosists of multiple components, some of the more significant for training and segmentation purposes include: histogram equalization, gaussian thresholding, a min/max normalization function, a custom function to extract subvolumes of a desired size, a custom dataloader with a transform arg that is currently set to take custom class "Transformations" which randomly flips and rotates subvolumes and labels (this can be disabled by setting "transform=None"), a U-Net3D-like model that takes input arg n_filters to determine how large the model should be, a 3D IoU loss function, a 3D Dice loss function, the pytorch BCELoss function, a custom trainning and validation cell, a custom testing cell, a custom mask generation cell, and a cell for displaying the generated mask onto the raw input data. The file, "final_trained_synapse_model.pth", is a trained state dictionary that can be used for mask generation.

All of the hyperparameters for training, and mask generation are held in the second cell of the notebook with annotations of what they do. It is essential for the functon of mask generation and testing that these parameters are the same as they were for training. The current values are equal to that of which "final_trained_synapse_model.pth" was trained on.

To run mask generation using a the pre-trained model, you must first run cells 1-11, next, run cell 19 so that the model can be instantiated, then you can go down to and run cell 24 to generate your mask. After "Finished Mask Generation" is printed, the mask can be displayed alongside the raw data by running cell 25. This final cell will instantiate a local server running neuroglancer for displaying the volumetric data (you will need to set the "PORT" parameter to an unused port number it is 9000 by default).
