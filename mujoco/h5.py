import h5py
file_path = "/home/daikiyotsufuji/.d4rl/datasets/halfcheetah_random-v2.hdf5"
try:
    with h5py.File(file_path, "r") as f:
        print(f.keys())
except Exception as e:
    print(f"Error reading HDF5 file: {e}")
