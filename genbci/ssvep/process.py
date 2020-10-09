import numpy as np
import mne
import scipy.io as sio

filename = "./data/session2.bin"
sFreq = 500.0
a = np.fromfile(filename, dtype=np.float32)
nChans = 32
nTimesteps = len(a) // nChans
a = np.reshape(a, (nTimesteps, nChans)).T


startTime = 7.75
startTimeIdx = int(np.floor(startTime * sFreq))
endTime = 324.82
endTimeIdx = int(np.floor(endTime * sFreq))
a = a[:, startTimeIdx : endTimeIdx + 1]

# for chan in range(a.shape[0]):
#     a[chan, :] = a[chan, :] / np.mean(a[chan, :]) - np.std(a[chan, :])


nTrials = 40
trialLength = 5
actualLength = 5
restLength = 3

print(a.shape[1])
nTimesteps = ((trialLength + restLength) * (nTrials - 1) + trialLength) * sFreq
print(nTimesteps)

trials = np.zeros((nTrials, 32, int(actualLength * sFreq)))

for i in range(nTrials):

    idx = int((i * trialLength + i * restLength) * sFreq)
    trials[i, :, :] = a[
        :, idx : int(idx + (actualLength * sFreq)),
    ]


channel_types = ["eeg"] * 32
info = mne.create_info(nChans, sFreq, channel_types)


trialLabels = [
    6,
    6,
    6,
    6,
    4,
    8,
    2,
    2,
    8,
    6,
    8,
    8,
    2,
    4,
    8,
    4,
    2,
    6,
    2,
    8,
    4,
    2,
    8,
    4,
    6,
    6,
    6,
    2,
    4,
    2,
    8,
    6,
    4,
    8,
    2,
    8,
    2,
    4,
    4,
    4,
]

events = np.array([[i, 0, trialLabels[i]] for i in range(nTrials)])
event_id = dict(_6=2, _8=4, _10=6, _15=8)
epochs = mne.EpochsArray(trials, info, events, event_id=event_id)


_ = epochs["_6"].plot(time_unit="s")
epochs["_6"].plot_psd()


sio.savemat("./session2.mat", {"X": epochs.get_data(), "y": trialLabels})

# eeg_evoked = epochs.average()


### Artifact removal
# ica = mne.preprocessing.ICA()
# ica.fit(epochs)
# ica.plot_overlay(eeg_evoked, exclude=[1, 2], picks="eeg")

# # ica.exclude = [14, 15, 16, 22, 23, 24, 25, 26, 27]
# ica.plot_sources(epochs)

# reconst_epochs = epochs.copy()
# ica.apply(reconst_epochs)


# reconst_epochs["_8"].plot_psd()

# ica.plot_properties(epochs, picks=[0, 1])


# frequencies = np.arange(1, 20, 1)
# power = mne.time_frequency.tfr_multitaper(
#     epochs, n_cycles=2, return_itc=False, freqs=frequencies, decim=3
# )
# power.plot([20])


# startTime = 5
# startTimeIdx = int(np.floor(startTime * sfreq))
# endTime = 35
# endTimeIdx = int(np.floor(endTime * sfreq))

# a = a[:, startTimeIdx:endTimeIdx]

# for chan in range(a.shape[0]):
#     a[chan, :] = a[chan, :] / np.mean(a[chan, :]) - np.std(a[chan, :])

# ch_types = ["eeg"] * 32
# ch_names = [f"eeg{i}" for i in range(1, 33)]

# info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
# raw = mne.io.RawArray(a, info)
# raw.filter(None, 50.0, fir_design="firwin")
# picks = mne.pick_types(
#     raw.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads"
# )
# raw.plot_psd(0, 50, area_mode="range", tmax=10.0, picks=picks, average=False)
# scalings = "auto"  # Could also pass a dictionary with some value == 'auto'
# raw.plot(
#     n_channels=4,
#     scalings=scalings,
#     title="Auto-scaled Data from arrays",
#     show=True,
#     block=True,
# )

# mne.viz.plot_raw_psd(raw, 0, 50)


# events = np.array([[0, a.shape[1], 1],])

# epochs = mne.EpochsArray(a[None, :, :], info, events, 0)
# frequencies = np.arange(1, 20, 1)
# power = mne.time_frequency.tfr_multitaper(
#     epochs, n_cycles=2, return_itc=False, freqs=frequencies, decim=3
# )
# power.plot(dB=True)

