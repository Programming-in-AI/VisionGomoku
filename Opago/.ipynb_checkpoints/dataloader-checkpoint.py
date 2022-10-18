import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import os
from tqdm import tqdm

def Dataloader(data_path):
    w, h = 15, 15
    base_path = os.path.join(data_path, '*.npz')

    file_list = glob(base_path)
    x_data, y_data = [], []
    for file_path in tqdm(file_list):
        data = np.load(file_path)
        x_data.extend(data['inputs'])
        y_data.extend(data['outputs'])

    x_data = np.array(x_data, np.float32).reshape((-1, h, w, 1))
    y_data = np.array(y_data, np.float32).reshape((-1, h * w))

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2020)

    del x_data, y_data

    print('x_train_shape: ', x_train.shape, 'y_train_shape: ', y_train.shape)
    print('x_val shape: ', x_val.shape, 'y_val shape: ', y_val.shape)

    return x_train, x_val, y_train, y_val
