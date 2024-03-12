import torch
from torch.utls.data import DataLoader
import numpy as np
import os


def convert_to_windows(data, n_window):
    """
    Split timeseries data into n_windows windows
    """
    windows = list(torch.split(data, n_window))
    for i in range (n_window-windows[-1].shape[0]):
        windows[-1] = torch.cat((windows[-1], windows[-1][-1].unsqueeze(0)))
    return torch.stack(windows)

def load_dataset(dataset, part=None):
    """
    Input: File Path
    Output: Training, Test and Validation sets
            with Labels for anomalous results in test and validation
    """
    loader = [] 
    folder = dataset

    for file in ['train', 'test', 'validation', 'labels', 'labels_validation']:
        if part is None:
            loader.append(np.load(os.path.join(folder, f'{file}.npy')))
        else:
            loader.append(np.load(os.path.join(folder, f'{part}_{file}.npy')))
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    validation_loader = DataLoader(loader[2], batch_size=loader[2].shape[0])
    return train_loader, test_loader, validation_loader, loader[3], loader[4]
