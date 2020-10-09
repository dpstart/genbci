import numpy as np
import mne
import scipy.io as sio

filename = "./data/session3.bin"
sFreq = 500.0
a = np.fromfile(filename, dtype=np.float32)
nChans = 32
nTimesteps = len(a) // nChans
a = np.reshape(a, (nTimesteps, nChans)).T


startTime = 7.02
startTimeIdx = int(np.floor(startTime * sFreq))
endTime = 323.80
endTimeIdx = int(np.floor(endTime * sFreq))
a = a[:, startTimeIdx : endTimeIdx + 1]

for chan in range(a.shape[0]):
    a[chan, :] = a[chan, :] / np.mean(a[chan, :]) - np.std(a[chan, :])


nTrials = 40
trialLength = 5
restLength = 3

print(a.shape[1])
nTimesteps = ((trialLength + restLength) * (nTrials - 1) + trialLength) * sFreq
print(nTimesteps)

trials = np.zeros((nTrials - 1, 32, int(trialLength * sFreq)))

for i in range(nTrials - 1):

    idx = int((i * trialLength + i * restLength) * sFreq)
    trials[i, :, :] = a[:, idx : int(idx + (trialLength * sFreq))]


channel_types = ["eeg"] * 32
info = mne.create_info(nChans, sFreq, channel_types)


trialLabels = [
    6,
    6,
    2,
    2,
    4,
    2,
    6,
    4,
    2,
    8,
    8,
    4,
    2,
    2,
    6,
    8,
    8,
    2,
    6,
    8,
    8,
    8,
    8,
    4,
    4,
    2,
    4,
    4,
    6,
    4,
    6,
    4,
    2,
    6,
    4,
    8,
    8,
    6,
    2,
    6,
]

events = np.array([[i, 0, trialLabels[i]] for i in range(nTrials - 1)])
event_id = dict(_6=2, _8=4, _10=6, _15=8)
epochs = mne.EpochsArray(trials, info, events, event_id=event_id)
epochs.filter(None, h_freq=40, filter_length=250)


sio.savemat("./session3.mat", {"X": epochs.get_data(), "y": trialLabels})

