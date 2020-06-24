import mne

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

from scipy.signal import lfilter

from dataclasses import dataclass
from typing import Optional, Sequence
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter

from genbci.util import compute_harmonics, compute_subharmonics


@dataclass
class EvokedFrequency:

    psd: np.ndarray
    snr: np.ndarray
    frequencies: Sequence
    tfr: Optional[np.array] = None


class SSVEP(mne.Epochs):
    def __init__(
        self,
        epochs: mne.Epochs,
        stimulation_freq: Sequence,
        psd: np.array = None,
        freqs: np.ndarray = None,
        harmonics=range(2, 5),
        subharmonics=range(2, 5),
        fmin: float = 0.1,
        fmax: float = 50,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        noisebandwidth: float = 1.0,
        compute_tfr: bool = False,
        tfr_method="rls",
        tfr_time_window=0.9,
    ):

        self.epochs = epochs
        self.info = deepcopy(epochs.info)

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
        self.snr = self.get_snr_single(self.frequencies)

        self.stimulation = EvokedFrequency(
            psd=self.get_amp(self.stimulation_frequencies),
            snr=self.get_snr_single(self.stimulation_frequencies),
            frequencies=self.stimulation_frequencies,
            tfr=(
                self._compute_tfr(
                    epochs, self.stimulation_frequencies, window_width=tfr_time_window
                )
                if compute_tfr
                else None
            ),
        )

        ###### Harmonics
        harm_freqs, harm_suborder = compute_harmonics(
            harmonics, fmin=self.fmin, fmax=self.fmax
        )

        self.harmonic = EvokedFrequency(
            frequencies=harm_freqs,
            psd=self.get_amp(harm_freqs),
            snr=self.get_snr_single(harm_freqs),
            tfr=(
                self._compute_tfr(epochs, harm_freqs, window_width=tfr_time_window)
                if compute_tfr
                else None
            ),
        )

        ###### Subharmonics
        sub_freqs, sub_suborder = compute_subharmonics(
            subharmonics, fmin=self.fmin, fmax=self.fmax
        )
        self.subharmonic = EvokedFrequency(
            frequencies=sub_freqs,
            psd=self.get_amp(sub_freqs),
            snr=self.get_snr_single(sub_freqs),
            tfr=(
                self._compute_tfr(epochs, sub_freqs, window_width=tfr_time_window)
                if compute_tfr
                else None
            ),
        )

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

    def get_amp(self, freqs):
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

    def get_snr(self, collapse_epochs=True, collapse_electrodes=True):
        # Construct the SNR spectrum
        ydata = np.stack(
            [self.get_snr_single(freq) for idx, freq in enumerate(self.frequencies)],
            axis=-1,
        )
        # Average over axes if necessary
        ydata = ydata.mean(
            axis=tuple(
                [x for x in range(2) if [collapse_epochs, collapse_electrodes][x]]
            )
        )

        return ydata

    # Helper functions to get specific frequencies:
    def get_snr_single(self, freqs):
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


from sklearn.base import BaseEstimator, TransformerMixin
from genbci.util import butter_bandpass
from pyriemann.utils.covariance import covariances


class SSVEPCovariances(BaseEstimator, TransformerMixin):
    """Estimate special form covariance matrix for SSVEP.
    Estimation of special form covariance matrix dedicated to SSVEP processing.
    A frequency banks is applied on the signal :math:`\mathbf{X}_i`
    and a super trial is build using the concatenation of :math:`f=1, \ldots, F` frequency bands :
    .. math::
        \mathbf{\\tilde{X}}_i =  \left[
                                 \\begin{array}{c}
                                 \mathbf{X}_{i,1} \\\\
                                 \vdots \\\\
                                 \mathbf{X}_{i,F} \\\\
                                 \end{array}
                                 \\right]
    This super trial :math:`\mathbf{\\tilde{X}}_i` will be used for covariance
    estimation.
    This allows to take into account the frequency structure of the signal, as
    described in [1,2,3].
    Parameters
    ----------
    estimator : string (default: 'scm')
        covariance matrix estimator. For regularization consider 'lwf' or 'oas'
        For a complete list of estimator, see `utils.covariance`.
    freq_targets : list of float
        list of n_frequency target frequencies.
    freq_band : float
        half-width of the frequency bands.
    fs : float
        sampling frequency of the signal.
    order : integer (default: 4)
        order of the Butterworth filter.
    filttype : string (default: 'forward')
        type of the Butterworth filter: 'forward' or 
        'forward-backward' for a bilateral filtering.
    See Also
    --------
    Covariances
    CospCovariances
    References
    ----------
    [1] M. Congedo, A. Barachant, A. Andreev ,"A New generation of
    Brain-Computer Interface Based on Riemannian Geometry", arXiv: 1310.8115.
    2013.
    [2] TODO, NeuroComputing2016
    [3] TODO, NeuroInformatics2020
    """

    def __init__(
        self,
        estimator="scm",
        freq_targets=[13, 17, 21],
        freq_band=0.1,
        fs=256,
        order=4,
        filttype="forward-backward",
    ):
        """Init."""
        self.estimator = estimator
        self.freq_targets = freq_targets
        self.freq_band = freq_band
        self.fs = fs
        self.order = order
        self.filttype = filttype

    def fit(self, X, y):
        """Fit.
        Do nothing. For compatibility purpose.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial.
        Returns
        -------
        self : SSVEPCovariances instance
            The SSVEPCovariances instance.
        """

        return self

    def transform(self, X):
        """Estimate SSVEP form covariance matrices.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of raw trials.
        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channel*n_frequency, n_channel*n_frequency)
            ndarray of SSVEP covariance matrices for each trials.
        """

        X_ext = []
        for f in self.freq_targets:
            X_ext.append(
                butter_bandpass(
                    X,
                    lowcut=f - self.freq_band,
                    highcut=f + self.freq_band,
                    fs=self.fs,
                    order=self.order,
                    filttype=self.filttype,
                )
            )
        X_ext = np.concatenate(X_ext, axis=1)
        covmats = covariances(X_ext, estimator=self.estimator)
        return covmats
