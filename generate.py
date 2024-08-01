import argparse
import torch
import numpy as np
import h5py
import time
import neuroglancer
from utils.data_preprocessing import normalize
from utils.constants import (
    SAMPLE_SHAPE, THRESHOLD, NUM_FILTERS, PORT
)
from model import SynapseSegmentationModel  # Update this import with the correct path to your model

def main(data_path, model_path):
    # Load the volume and labels from the data file
    with h5py.File(data_path, 'r') as f:
        volume = f['volumes/raw'][:]
        labels = f['volumes/labels/clefts'][:]

    # Convert files to float
    labels = labels.astype(np.float32)
    volume = volume.astype(np.uint8)
    original_volume = volume

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('Cache cleared')
    print(device)

    # Initialize the model
    trained_model = SynapseSegmentationModel(n_filters=NUM_FILTERS)

    # Load the saved state dictionary
    state_dict = torch.load(model_path)
    trained_model.load_state_dict(state_dict)
    trained_model = trained_model.to(device)  # Make model compatible with CUDA
    trained_model.eval()

    # Set patch size
    patch_size = SAMPLE_SHAPE

    # Initialize segmentation mask
    segmentation_mask = np.zeros(volume.shape, dtype=np.uint8)

    print('Generating Mask...')

    # Iterate over the large array and get a patch
    for z in range(0, volume.shape[0], patch_size[0]):
        for y in range(0, volume.shape[1], patch_size[1]):
            for x in range(0, volume.shape[2], patch_size[2]):
                patch = volume[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                if patch.shape == tuple(patch_size):  # Check that patch is the correct shape
                    # Pre-process the patch
                    patch = normalize(patch)

                    # Convert the patch to a tensor and add a batch dimension
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)

                    # Perform segmentation inference on the patch
                    with torch.no_grad():
                        prediction = trained_model(patch_tensor.float())

                    # Post-process
                    prediction[prediction < THRESHOLD] = 0
                    prediction[prediction >= THRESHOLD] = 1

                    # Convert prediction to a numpy array
                    prediction = prediction.squeeze().cpu().detach().numpy()

                    # Reconstruct segmentation mask
                    segmentation_mask[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] = prediction

    print('Finished Mask Generation')

    print('\nSaving mask...')

    # Save the mask as an h5py file
    with h5py.File('segmentation_mask.hdf', 'w') as f:
        f.create_dataset('data', data=segmentation_mask)

    print('Save Complete')

    # View segmented mask on the raw data
    ip = 'localhost'  # Or public IP of the machine for sharable display
    port = PORT  # Ensure PORT is defined somewhere in your script or config
    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    viewer = neuroglancer.Viewer()

    # Load the raw image data
    raw = original_volume

    print('Loading mask...')
    # Load the segmentation mask
    with h5py.File('segmentation_mask.hdf', 'r') as f:
        seg = f['data'][:]

    # Make sure the mask and raw data have the same shape 
    print('\nRaw Image Shape: ', raw.shape, '\nMask Shape: ', seg.shape, '\n')

    res = neuroglancer.CoordinateSpace(
        names=['z', 'y', 'x'],
        units=['nm', 'nm', 'nm'],
        scales=[1, 1, 1]
    )

    def ngLayer(data, res, oo=[0, 0, 0], tt='segmentation'):
        return neuroglancer.LocalVolume(data, dimensions=res, volume_type=tt, voxel_offset=oo)

    print('Setting up Neuroglancer viewer...')
    with viewer.txn() as s:
        s.layers.append(name='raw', layer=ngLayer(raw, res, tt='image'))
        s.layers.append(name='seg', layer=ngLayer(seg, res, tt='segmentation'))
    print('Neuroglancer viewer setup complete.')

    print('Viewer URL:', viewer)
    print('You can view the data at the provided URL in your web browser.')

    # Keep the script running to keep the viewer active
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Viewer closed by user.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run segmentation model and view results.')
    parser.add_argument('data_path', type=str, help='Path to the input data file')
    parser.add_argument('model_path', type=str, help='Path to the saved model state dictionary')
    args = parser.parse_args()
    
    main(args.data_path, args.model_path)
