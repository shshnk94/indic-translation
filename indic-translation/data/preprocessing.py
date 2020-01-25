import pandas as pd
import numpy as np

def split_data(file_path, destination):

    data = pd.read_csv(file_path, sep='\t', header=None)[[0, 1]]

    mask = np.random.rand(data.shape[0]) < 0.8
    train = data[mask]
    valid = data[~mask]

    train.to_csv(destination + 'train', header=False, index=False)
    valid.to_csv(destination + 'valid', header=False, index=False)
