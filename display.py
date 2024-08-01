import h5py
import neuroglancer
from utils.constants import PORT
import time

def main(data_path, mask_path):

    print('Loading raw image data...')
    # Load raw image data
    with h5py.File(data_path, 'r') as f:
        raw = f['volumes/raw'][:]
    print('Raw image data loaded.')

    print('Loading segmentation mask...')
    # Load the segmentation mask from the provided path
    with h5py.File(mask_path, 'r') as f:
        seg = f['data'][:]
    print('Segmentation mask loaded.')

    # View segmented mask on the raw data
    ip = 'localhost'  # Or public IP of the machine for sharable display
    port = PORT  # Ensure PORT is defined somewhere in your script or config
    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    viewer = neuroglancer.Viewer()

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
    import sys
    if len(sys.argv) != 3:
        print("Usage: python app.py <path_to_data> <path_to_mask>")
        sys.exit(1)
    data_path = sys.argv[1]
    mask_path = sys.argv[2]
    main(data_path, mask_path)
