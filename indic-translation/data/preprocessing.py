import pandas as pd
import numpy as np

def preprocessing(language='hin'):

    data = pd.read_csv(language + '.txt', sep='\t', header=None)[[0, 1]]

    mask = np.random.rand(data.shape[0]) < 0.8
    train = data[mask]
    valid = data[~mask]

    train.to_csv('train', header=False, index=False)
    valid.to_csv('valid', header=False, index=False)
