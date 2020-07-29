# coding=utf-8
from torch.autograd import Variable
from torch.nn import Module
import torch
import torch.nn as nn

import numpy as np
import scipy.io as sio
from scipy.signal import filtfilt, butter
import mne

import random
import gzip
import pickle
import os


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def init_torch_and_get_device(random_state=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    random.seed(random_state)

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_harmonics(frequencies, fmin=0.1, fmax=50, orders=range(2, 5)):
    if not orders:
        return None, None

    if type(orders) is bool:
        orders = [o for o in range(2, 6)]
    elif type(orders) is int:
        orders = [o for o in range(2, orders)]
    try:
        freqs = [
            [n * f for n in orders if n * f <= fmax and n * f >= fmin]
            for f in frequencies
        ]
    except:
        freqs = [
            [n * f for n in orders if n * f <= fmax and n * f >= fmin]
            for f in [frequencies]
        ]

    return np.array(freqs), np.array(orders)


def compute_subharmonics(frequencies, fmin=0.1, fmax=50, orders=range(2, 5)):
    if not orders:
        return None, None

    if type(orders) is bool:
        orders = [o for o in range(2, 6)]  # default: 4 harmonics
    elif type(orders) is int:
        orders = [o for o in range(2, orders)]

    freqs = [
        [f / o if (f / o <= fmax and f / o >= fmin) else np.nan for o in orders]
        for f in frequencies
    ]
    # freqs = [f for sublist in freqs for f in sublist]

    return np.array(freqs), np.array(orders)


def get_balanced_batches(n_trials, rng, shuffle, n_batches=None, batch_size=None):
    """Create indices for batches balanced in size 
    (batches will have maximum size difference of 1).
    Supply either batch size or number of batches. Resulting batches
    will not have the given batch size but rather the next largest batch size
    that allows to split the set into balanced batches (maximum size difference 1).
    Parameters
    ----------
    n_trials : int
        Size of set.
    rng : RandomState
    shuffle : bool
        Whether to shuffle indices before splitting set.
    n_batches : int, optional
    batch_size : int, optional
    Returns
    -------
    """
    assert batch_size is not None or n_batches is not None
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))

    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches
    all_inds = np.array(range(n_trials))
    if shuffle:
        rng.shuffle(all_inds)
    i_start_trial = 0
    i_stop_trial = 0
    batches = []
    for i_batch in range(n_batches):
        i_stop_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            i_stop_trial += 1
        batch_inds = all_inds[range(i_start_trial, i_stop_trial)]
        batches.append(batch_inds)
        i_start_trial = i_stop_trial
    assert i_start_trial == n_trials
    return batches


def get_exo_data(PATH: str, plot=False) -> mne.Epochs:

    event_id = dict(resting=1, stim13=2, stim17=3, stim21=4)
    tmin, tmax = 2.0, 5.0

    files = [
        "./subject12/record-[2014.03.10-19.47.49]",
        "./subject01/record-[2012.07.06-19.02.16]",
    ]

    all_epochs = None

    for file in files:

        full_path = os.path.join(PATH, file)

        raw = mne.io.read_raw_fif(str(full_path) + "_raw.fif", preload=True)
        events = mne.read_events(str(full_path) + "-eve.fif")

        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
        raw.filter(6.0, 30.0, method="iir", picks=picks)

        if plot is True:
            raw.plot(
                events=events,
                event_color={1: "red", 2: "blue", 3: "green", 4: "cyan"},
                duration=6,
                n_channels=8,
                color={"eeg": "steelblue"},
                scalings={"eeg": 2e-2},
                show_options=False,
                title="Raw EEG from S12",
            )

        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            tmin,
            tmax,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
            verbose=False,
        )

        if plot is True:
            epochs.plot(title="SSVEP epochs", n_channels=8, n_epochs=4)

        all_epochs = (
            epochs
            if all_epochs is None
            else mne.concatenate_epochs([all_epochs, epochs])
        )

    return all_epochs


def get_data(subject, training, PATH):
    """	Loads the dataset 2a of the BCI Competition IV
    available on http://bnci-horizon-2020.eu/database/data-sets
    Keyword arguments:
    subject -- number of subject in [1, .. ,9]
    training -- if True, load training data
                if False, load testing data
    
    Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
            class_return 	numpy matrix 	size = NO_valid_trial
    """
    NO_channels = 22
    NO_tests = 6 * 48
    Window_Length = 7 * 250

    class_return = np.zeros(NO_tests)
    data_return = np.zeros((NO_tests, NO_channels, Window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(PATH + "A0" + str(subject) + "T.mat")
    else:
        a = sio.loadmat(PATH + "A0" + str(subject) + "E.mat")
    a_data = a["data"]
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_fs = a_data3[3]
        a_classes = a_data3[4]
        a_artifacts = a_data3[5]
        a_gender = a_data3[6]
        a_age = a_data3[7]
        for trial in range(0, a_trial.size):
            if a_artifacts[trial] == 0:
                data_return[NO_valid_trial, :, :] = np.transpose(
                    a_X[
                        int(a_trial[trial]) : (int(a_trial[trial]) + Window_Length), :22
                    ]
                )
                class_return[NO_valid_trial] = int(a_y[trial])
                NO_valid_trial += 1

    return data_return[0:NO_valid_trial, :, :], class_return[0:NO_valid_trial]


def cuda_check(module_list):
    """
    Checks if any module or variable in a list has cuda() true and if so
    moves complete list to cuda

    Parameters
    ----------
    module_list : list
        List of modules/variables

    Returns
    -------
    module_list_new : list
        Modules from module_list all moved to the same device
    """
    cuda = False
    for mod in module_list:
        if isinstance(mod, Variable):
            cuda = mod.is_cuda
        elif isinstance(mod, Module):
            cuda = next(mod.parameters()).is_cuda

        if cuda:
            break
    if not cuda:
        return module_list

    module_list_new = []
    for mod in module_list:
        module_list_new.append(mod.cuda())
    return module_list_new


def change_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find("MultiConv") != -1:
        for conv in m.convs:
            conv.weight.data.normal_(0.0, 1.0)
            if conv.bias is not None:
                conv.bias.data.fill_(0.0)
    elif classname.find("Conv") != -1 or classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 1.0)  # From progressive GAN paper
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif classname.find("BatchNorm") != -1 or classname.find("LayerNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


def butter_bandpass(signal, lowcut, highcut, fs, order=4, filttype="forward-backward"):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    if filttype == "forward":
        filtered = lfilter(b, a, signal, axis=-1)
    elif filttype == "forward-backward":
        filtered = filtfilt(b, a, signal, axis=-1)
    else:
        raise ValueError("Unknown filttype:", filttype)
    return filtered


def mne_epochs_from_simbci(path: str):

    mat_contents = sio.loadmat(path)
    mat_contents = mat_contents["dataset"]

    X = mat_contents["X"][0][0]
    Fs = mat_contents["samplingFreq"][0][0][0][0]
    freqs = mat_contents["freqs"][0][0][0]

    # Events in the form name, latency, duration
    events = []
    for elem in mat_contents["events"][0][0][0]:
        events.append([elem[0][0], elem[1][0][0], elem[2][0][0]])

    # TODO no hardcoding
    events_id = {"Freq1": 1, "Freq2": 2, "Freq3": 3}
    filtered = list(
        filter(lambda x: x[0] == "Freq1" or x[0] == "Freq2" or x[0] == "Freq3", events)
    )

    events_mne = [[x[1], 0, events_id[x[0]]] for x in filtered]

    ch_names = ["ch" + str(i) for i in range(X.shape[1])]
    ch_types = ["eeg"] * X.shape[1]
    info = mne.create_info(ch_names=ch_names, sfreq=Fs, ch_types=ch_types)

    trials = 18
    X = X[2000:, :]

    X = np.reshape(X, (trials, -1, X.shape[1]))  # epochs, samples, channels
    X = np.transpose(X, (0, 2, 1))  # epochsxchannelsxsamples

    return mne.EpochsArray(X, info=info, events=events_mne, event_id=events_id)
