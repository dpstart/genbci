import numpy as np
import mne
import scipy.io as sio

filename = "./data/single.bin"
sFreq = 500.0
a = np.fromfile(filename, dtype=np.float32)
nChans = 32
nTimesteps = len(a) // nChans
a = np.reshape(a, (nTimesteps, nChans)).T


startTime = 5.13
startTimeIdx = int(np.floor(startTime * sFreq))
endTime = 35.05
endTimeIdx = int(np.floor(endTime * sFreq))
a = a[:, startTimeIdx : endTimeIdx + 1]

for chan in range(a.shape[0]):
    a[chan, :] = a[chan, :] / np.mean(a[chan, :]) - np.std(a[chan, :])


channel_types = ["eeg"] * 32
info = mne.create_info(nChans, sFreq, channel_types)

raw = mne.io.RawArray(a, info)
raw.filter(None, 50.0, fir_design="firwin")
picks = mne.pick_types(
    raw.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads"
)
raw.plot_psd()

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

