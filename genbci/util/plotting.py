import matplotlib.pyplot as plt
import numpy as np


def plot_snr(snr, frequencies):
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
    _plot_spectrum(snr, frequencies)


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
