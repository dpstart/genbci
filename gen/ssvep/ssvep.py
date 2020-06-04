import mne

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

from scipy.signal import lfilter

from dataclasses import dataclass
from typing import Optional, Sequence
from copy import deepcopy
import xarray as xr
from scipy.ndimage.filters import gaussian_filter

@dataclass
class EvokedFrequency:

    psd: np.ndarray
    snr: np.ndarray
    frequencies: Sequence
    tfr: Optional[xr.DataArray] = None


class SSVEP(mne.Epochs):
    def __init__(
        self,
        epochs: mne.Epochs,
        stimulation_freq: Sequence,
        psd: np.array = None,
        freqs: np.ndarray = None,
        fmin: float = 0.1,
        fmax: float = 50,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        noisebandwidth: float = 1.0,
        compute_tfr: bool = False,
        tfr_method="rls",
        tfr_time_window=0.9,
    ):

        self.info = deepcopy(epochs.info)
        self.events = deepcopy(epochs.events)

        self.frequencies = freqs
        self.psd = psd
        self.fmin = fmin
        self.fmax = fmax
        self.noisebandwidth = noisebandwidth

        if self.psd is None:
            self.psd, self.frequencies = mne.time_frequency.psd_multitaper(
                epochs, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax
            )

        self.frequency_resolution = self.frequencies[1] - self.frequencies[0]
        self.stimulation_frequencies = np.array(stimulation_freq, dtype=float)

        self.snr = self._get_snr(self.frequencies)
        self.stimulation = EvokedFrequency(
            psd=self._get_amp(self.stimulation_frequencies),
            snr=self._get_snr(self.stimulation_frequencies),
            frequencies=self.stimulation_frequencies,
            tfr=(
                self._compute_tfr(
                    epochs, self.stimulation_frequencies, window_width=tfr_time_window
                )
                if compute_tfr
                else None
            ),
        )
        # TODO Add harmonics

    def predict_epochs(
        self,
        labels,
        trainepochs=None,
        method=sklearn.linear_model.LogisticRegressionCV(),
    ):
        labels = np.array(labels)

        assert labels.size == len(
            self
        ), "Number of labels must match number \
        of epochs"

        X = self.stimulation.snr.reshape((len(self), -1))

        # TODO Add harmonics
        # if the training trials
        if trainepochs is not None:
            trainepochs = np.array(trainepochs)
            if trainepochs.size == len(self):
                # probably a boolean
                testepochs = ~trainepochs
            else:
                # probably integer index
                testepochs = np.array(
                    [i for i in range(len(self)) if i not in trainepochs]
                )
        else:
            # use all epochs for both training and test
            trainepochs = range(len(self))
            testepochs = range(len(self))

        # train the model
        method.fit(X[trainepochs, :], labels[trainepochs])
        # predict the model
        predictions = method.predict(X[testepochs, :])

        return np.array(predictions)

    def _get_amp(self, freqs):
        """
        Helper function to get the freq-smoothed amplitude of a frequency
        """
        return np.stack(
            [
                self.psd[
                    ...,
                    (
                        (self.frequencies <= freq + self.frequency_resolution)
                        & (self.frequencies >= freq - self.frequency_resolution)
                    ),
                ].mean(axis=-1)
                for freq in freqs.flat
            ],
            axis=-1,
        )

    # Helper functions to get specific frequencies:
    def _get_snr(self, freqs):
        """
        Helper function to work out the SNR of a given frequency
        """
        snr = []
        for freq in freqs.flat:
            stimband = (self.frequencies <= freq + self.frequency_resolution) & (
                self.frequencies >= freq - self.frequency_resolution
            )
            noiseband = (
                (self.frequencies <= freq + self.noisebandwidth)
                & (self.frequencies >= freq - self.noisebandwidth)
                & ~stimband
            )
            snr.append(
                self.psd[..., stimband].mean(axis=-1)
                / self.psd[..., noiseband].mean(axis=-1)
            )
        snr = np.stack(snr, axis=-1) if len(snr) > 1 else snr[0]
        return snr

    def _compute_tfr(
        self, epoch, freq, tfr_method="rls", window_width=1.2, filter_lambda=1.0
    ):
        """
        Work out the time-frequency composition of different frequencies.
        """

        data = epoch.get_data()

        if type(freq) is not np.ndarray:
            raise TypeError("Frequencies need to provided in a numpy array.")

        samplefreq = epoch.info["sfreq"]
        n_window = int(samplefreq * window_width)
        if filter_lambda == 1:
            lambdafilter = np.ones(n_window) / (n_window / 2)
        else:
            lambdafilter = np.power(filter_lambda, np.arange(n_window))

        t = np.arange(data.shape[-1]) / samplefreq

        # create a data structure that matches MNE standard TFR shape
        tfr_data = np.zeros((data.shape[0], data.shape[1], freq.size, data.shape[2]))

        if tfr_method == "rls":
            # this implementation follows Sucharit Katyal's code
            for fi, f in enumerate(freq.flatten()):
                s = -np.sin(2 * np.pi * f * t)
                c = np.cos(2 * np.pi * f * t)
                # Create the sin and cosine
                for trial in range(data.shape[0]):
                    for electrode in range(data.shape[1]):
                        y = data[trial, electrode, :]
                        # obtain cosine and since components
                        hc = lfilter(lambdafilter, 1, y * c)
                        hs = lfilter(lambdafilter, 1, y * s)
                        # lambda correction, if necessary
                        if filter_lambda < 1:
                            hc = hc / lfilter(lambdafilter, 1, c ** 2)
                            hs = hs / lfilter(lambdafilter, 1, s ** 2)
                        # combine the data to get envelope
                        a = np.abs(hc + 1j * hs)
                        # shift left, pad zero
                        a = np.roll(a, -n_window // 2)
                        a[(-n_window // 2) : -1] = np.nan
                        # smooth with gaussian
                        a[0 : (-n_window // 2)] = gaussian_filter(
                            a[0 : (-n_window // 2)], n_window // 10
                        )
                        # set in tfr_data
                        tfr_data[trial, electrode, fi, :] = a
        else:
            raise NotImplementedError("Only RLS is available so far.")
        return tfr_data

    def plot_snr(self, collapse_epochs=True, collapse_electrodes=False,
                 **kwargs):
        """
        Plot the signal-to-noise-ratio-spectrum that has been calculated for
        this data.

        Parameters:
            collapse_epochs : bool
                Whether you want to plot the average of all epochs (default),
                or each power-spectrum individually.
            collapse_electrodes : bool
                Whether you want to plot each electrode individually
                (default), or only the average of all electrodes.

        """

        # Construct the SNR spectrum
        ydata = np.stack([self._get_snr(freq)
                          for idx, freq in enumerate(self.frequencies)],
                         axis=-1)
        # Average over axes if necessary
        ydata = ydata.mean(axis=tuple([x for x in range(2)
                                       if [collapse_epochs,
                                           collapse_electrodes][x]]))
        self._plot_spectrum(ydata, **kwargs)

    def _plot_spectrum(self, ydata, figsize=(15, 7), show=True):
        """
        Helper function to plot different spectra
        """
        # Make sure frequency data is the first index
        ydata = np.transpose(
            ydata, axes=([ydata.shape.index(self.frequencies.size)] +
                         [dim for dim, _ in enumerate(ydata.shape)
                          if dim != ydata.shape.index(self.frequencies.size)])
        )
        # Start figure
        plt.figure(figsize=figsize)
        xmarks = np.concatenate([a.flatten() for a in
                                 [self.stimulation.frequencies,
                                  self.harmonic.frequencies,
                                  self.subharmonic.frequencies,
                                  self.intermodulation.frequencies]
                                 if np.any(a)]).tolist()
        # If we didn't collapse over epochs, split the data
        if ydata.ndim <= 2:
            plt.plot(self.frequencies, ydata, color='blue', alpha=0.3)
            if ydata.ndim > 1:
                plt.plot(self.frequencies, ydata.mean(axis=1), color='red')
            for xval in xmarks:
                plt.axvline(xval, linestyle='--', color='gray')
            plt.xticks(xmarks)
            plt.title('Average spectrum of all epochs')
        elif ydata.ndim > 2:
            ydatas = [ydata[:, idx, :] for idx in range(ydata.shape[1])]
            for idx, ydata in enumerate(ydatas):
                plt.subplot(np.ceil(np.sqrt(len(ydatas))),
                            np.ceil(len(ydatas) /
                                    np.ceil(np.sqrt(len(ydatas)))),
                            idx + 1)
                plt.plot(self.frequencies, ydata, color='blue', alpha=0.3)
                if ydata.ndim > 1:
                    plt.plot(self.frequencies, ydata.mean(axis=1), color='red')
                for xval in xmarks:
                    plt.axvline(xval, linestyle='--', color='gray')
                plt.xticks(xmarks)
                plt.title('Spectrum of epoch {n}'.format(n=idx + 1))

        if show:
            plt.show()

    def plot_tfr(
        self,
        frequency="stimulation",
        collapse_epochs=True,
        collapse_electrodes=False,
        figsize=(7, 5),
    ):
        """
        Plot the time-course of one of the evoked frequencies.

        Args
        ----
            frequency : str
                Which evoked frequency to plot. Either 'stimulation',
                'harmonic', 'subharmonic' or 'intermodulation'
            collapse_epochs : bool
                Whether to average over the epochs or not.
            collapse_electrodes : bool
                Whether to average over electrodes or not.
            figsize : tup
                Matplotlib figure size.
        """

        if frequency is None or frequency == "stimulation":
            y = self.stimulation.tfr
            z = self.stimulation.frequencies
        elif type(frequency) is str:
            y = self.__getattribute__(frequency).tfr
            z = self.__getattribute__(frequency).frequencies

        x = np.arange(y.shape[-1]) / self.info["sfreq"]

        collapse_axes = tuple(
            [ax for ax, b in enumerate([collapse_epochs, collapse_electrodes]) if b]
        )
        if len(collapse_axes) > 0:
            y = y.mean(axis=collapse_axes)
        # Make time the first dimension
        y = np.rollaxis(y, -1)
        # Make a figure (-1 is now freq. dimension)
        nplots = y.shape[-1]
        nrows = int(np.ceil(np.sqrt(nplots)))
        ncols = int(np.ceil(nplots / nrows))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        # y = np.squeeze(y)
        for idx in range(nplots):
            # Choose axes to plot in
            ax = axes.flatten()[idx] if nplots > 1 else axes
            # Plot the individual lines
            ax.plot(x, y[..., idx], color="blue", alpha=0.1)
            # Plot the mean of the data
            if y[..., idx].size > y.shape[0]:
                ax.plot(x, y[..., idx].mean(axis=-1))
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title(str(z.flatten()[idx]) + " Hz")

        plt.show()


    def plot_psd(
        self, log_scale=True, collapse_epochs=True, collapse_electrodes=False, **kwargs,
    ):
        """
        Plot the power-spectrum that has been calculated for this data.
        Parameters
        ----------
        log_scale : bool
            Yes or no
        collapse_epochs : bool
            Whether you want to plot the average of all epochs (default),
            or each power-spectrum individually.
        collapse_electrodes : bool
            Whether you want to plot each electrode individually
            (default), or only the average of all electrodes.
        """
        ydata = self.psd
        if collapse_electrodes:
            ydata = ydata.mean("channel")
        if collapse_epochs:
            ydata = ydata.mean("epoch")
        return self._plot_spectrum(ydata, **kwargs)

    def __repr__(self):
        outstring = (
            "ssvepy data structure based on epoched data.\n"
            "The stimulation frequency(s) is {stimfreq}.\n"
            "There are {nepoch} epochs.\n The power spectrum was "
            "evaluated over {nfreqs} frequencies ({fmin} Hz - "
            "{fmax} Hz).\n".format(
                stimfreq=self.stimulation_frequencies,
                nepoch=self.psd.shape[0],
                nfreqs=self.frequencies.size,
                fmin=self.frequencies.min(),
                fmax=self.frequencies.max(),
            )
        )
        return outstring

    def __len__(self):
        return self.psd.shape[0]

    def __getitem__(self, i):
        return self.psd[i, ...]

    def __iter__(self):
        for i in range(len(self)):
            yield self.psd[i, ...]
