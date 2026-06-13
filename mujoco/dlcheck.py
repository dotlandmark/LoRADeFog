import h5py
import gzip
import numpy as np
import zlib

file_path = "/home/daikiyotsufuji/.d4rl/datasets/halfcheetah_random-v2.hdf5"  # 実際のパスに置き換えてください
new_file_path = "/home/daikiyotsufuji/.d4rl/datasets/halfcheetah_random-v2_fixed.hdf5"

def decompress_zlib_data(dataset):
    """非同期で圧縮データを読み込み、zlibで解凍する"""
    decompressed_data = b""
    for chunk in dataset.iter_chunks():
        compressed_chunk = chunk[0]  # チャンクの最初の部分を取得
        decompressed_data += zlib.decompress(compressed_chunk)  # zlibで解凍
    return np.frombuffer(decompressed_data, dtype=dataset.dtype).reshape(dataset.shape)

with h5py.File(file_path, 'r') as f:
    with h5py.File(new_file_path, 'w') as new_f:
        for key in f.keys():
            if key == "actions":
                print(f"Attempting to decompress 'actions'")
                try:
                    dataset = f[key]
                    decompressed_data = decompress_zlib_data(dataset)
                    new_f.create_dataset(key, data=decompressed_data)
                    print(f"'actions' decompressed and saved successfully.")
                except Exception as e:
                    print(f"Error decompressing 'actions': {e}")
            else:
                f.copy(key, new_f)
