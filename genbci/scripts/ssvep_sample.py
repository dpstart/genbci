import numpy as np
from scipy import signal

import torch
import torch.utils.data

# pre-processing definition
def data_process(input_data):

    input_data = input_data.swapaxes(1, 2)
    #  referencing
    dataref = input_data[:, :, 0]
    for i in range(0, 2):
        input_data[:, :, i] = input_data[:, :, i] - dataref

    fs = 500
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    Q = 30.0  # Quality factor
    w0 = f0 / (fs / 2)  # Normalized Frequency

    # Design notch filter
    b, a = signal.iirnotch(w0, Q)
    for i in range(0, len(input_data)):
        input_data[i, :, :] = signal.filtfilt(
            b, a, input_data[i, :, :], axis=0
        )  # apply along the zeroeth dimension

    b, a = signal.butter(4, 9.0 / (fs / 2.0), "highpass")
    for i in range(0, len(input_data)):
        input_data[i, :, :] = signal.filtfilt(
            b, a, input_data[i, :, :], axis=0
        )  # apply along the zeroeth dimension

    b, a = signal.butter(4, 60 / (fs / 2.0), "lowpass")
    for i in range(0, len(input_data)):
        input_data[i, :, :] = signal.filtfilt(
            b, a, input_data[i, :, :], axis=0
        )  # apply along the zeroeth dimension

    min_data = np.min(input_data)
    range_data = np.max(input_data) - min_data
    input_data = (input_data - min_data) / range_data
    # input_data = input_data.swapaxes(1,2)

    return input_data


# data loading for training the generative model

PATH = "/home/paperspace/eegsourcegen/genbci"
# class0
data_class0 = [
    np.load(
        f"{PATH}/generate/data/ssvep/sampledata_class0.npy"
    )
]
data_class0 = np.concatenate(data_class0)
label_class0 = [
    np.load(
        f"{PATH}/generate/data/ssvep/samplelabel_class0.npy"
    )
]
label_class0 = np.concatenate(label_class0)
# pre-processing
data_class0 = data_process(data_class0)

# class1
data_class1 = [
    np.load(
        f"{PATH}/generate/data/ssvep/sampledata_class1.npy"
    )
]
data_class1 = np.concatenate(data_class1)
label_class1 = [
    np.load(
        f"{PATH}/generate/data/ssvep/samplelabel_class1.npy"
    )
]
label_class1 = np.concatenate(label_class1)
# pre-processing
data_class1 = data_process(data_class1)

# class2
data_class2 = [
    np.load(
        f"{PATH}/generate/data/ssvep/sampledata_class2.npy"
    )
]
data_class2 = np.concatenate(data_class2)
label_class2 = [
    np.load(
        f"{PATH}/generate/data/ssvep/samplelabel_class2.npy"
    )
]
label_class2 = np.concatenate(label_class2)
# pre-processing
data_class2 = data_process(data_class2)


for nclass in range(0, 3):

    if nclass == 0:
        data_train = data_class0
    if nclass == 1:
        data_train = data_class1
        label_train = label_class1
    if nclass == 2:
        data_train = data_class2
        label_train = label_class2

    data_train = data_train.swapaxes(1, 2)

datatrain = torch.from_numpy(data_train)
label = torch.from_numpy(label_train)

dataset = torch.utils.data.TensorDataset(datatrain, label)

