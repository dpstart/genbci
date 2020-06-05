import matplotlib.pyplot as plt
import numpy as np


def plot_snr(snr, frequencies, **kwargs):
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
    #    ydata = np.stack(
    # [self._get_snr(freq) for idx, freq in enumerate(self.frequencies)], axis=-1
    # )
    ## Average over axes if necessary
    # ydata = ydata.mean(
    # axis=tuple([x for x in range(2) if [collapse_epochs, collapse_electrodes][x]])
    # )
    _plot_spectrum(snr, frequencies, **kwargs)


def plot_psd(
    psd,
    frequencies,
    log_scale=True,
    collapse_epochs=True,
    collapse_electrodes=False,
    **kwargs,
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
    ydata = psd
    ydata = ydata.mean(
        axis=tuple([x for x in range(2) if [collapse_epochs, collapse_electrodes][x]])
    )
    return _plot_spectrum(ydata, frequencies, **kwargs)


def _plot_spectrum(ydata, frequencies, figsize=(15, 7), show=True):
    """
        Helper function to plot different spectra
        """
    # Make sure frequency data is the first index
    ydata = np.transpose(
        ydata,
        axes=(
            [ydata.shape.index(frequencies.size)]
            + [
                dim
                for dim, _ in enumerate(ydata.shape)
                if dim != ydata.shape.index(frequencies.size)
            ]
        ),
    )
    # Start figure
    plt.figure(figsize=figsize)
    # xmarks = np.concatenate(
    #    [
    #        a.flatten()
    #        for a in [
    #            self.stimulation.frequencies,
    #            self.harmonic.frequencies,
    #            self.subharmonic.frequencies,
    #            self.intermodulation.frequencies,
    #        ]
    #        if np.any(a)
    #    ]
    # ).tolist()

    # This should be fine for all paradigms
    xmarks = frequencies
    # If we didn't collapse over epochs, split the data
    if ydata.ndim <= 2:
        plt.plot(frequencies, ydata, color="blue", alpha=0.3)
        if ydata.ndim > 1:
            plt.plot(frequencies, ydata.mean(axis=1), color="red")
        # for xval in xmarks:
        #    plt.axvline(xval, linestyle="--", color="gray")
        plt.xticks(np.arange(int(frequencies[0]), int(frequencies[-1]), 5))
        plt.title("Average spectrum of all epochs")
    elif ydata.ndim > 2:
        ydatas = [ydata[:, idx, :] for idx in range(ydata.shape[1])]
        for idx, ydata in enumerate(ydatas):
            plt.subplot(
                np.ceil(np.sqrt(len(ydatas))),
                np.ceil(len(ydatas) / np.ceil(np.sqrt(len(ydatas)))),
                idx + 1,
            )
            plt.plot(frequencies, ydata, color="blue", alpha=0.3)
            if ydata.ndim > 1:
                plt.plot(frequencies, ydata.mean(axis=1), color="red")
            # for xval in xmarks:
            #    plt.axvline(xval, linestyle="--", color="gray")
            plt.xticks(np.arange(int(frequencies[0]), int(frequencies[-1]), 5))
            plt.title("Spectrum of epoch {n}".format(n=idx + 1))

    if show:
        plt.show()


def plot_tfr(
    tfr,
    frequencies,
    sfreq,
    collapse_epochs=True,
    collapse_electrodes=False,
    figsize=(7, 5),
):
    """
        Plot the time-course of one of the evoked frequencies.

        Args
        ----'
            collapse_epochs : bool
                Whether to average over the epochs or not.
            collapse_electrodes : bool
                Whether to average over electrodes or not.
            figsize : tup
                Matplotlib figure size.
        """

    # if frequency is None or frequency == "stimulation":
    #    y = self.stimulation.tfr
    #    z = self.stimulation.frequencies
    # elif type(frequency) is str:
    #    y = self.__getattribute__(frequency).tfr
    #    z = self.__getattribute__(frequency).frequencies

    y = tfr
    z = frequencies

    x = np.arange(y.shape[-1]) / sfreq

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
