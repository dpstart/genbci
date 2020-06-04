import time
from pylsl import StreamInfo, StreamOutlet

import sys; sys.path.append('..')

import numpy
from ssvep import SSVEP
from ssvep.example_data import epoch_example

# Create SSVEP object with sample data, train and predict
ssvep_example = SSVEP(epoch_example, [1.2, 6.0], fmin=0.5, fmax=30,
                      compute_tfr=True)

labels = (epoch_example.events[:, -1] // 100) - 1
predictions = ssvep_example.predict_epochs(labels)


number_classes = 4
number_channels = 1
stream_name = "Test4Class"
changeRate = (
    1  # number of samples before changing electrode when randomly changing electrodes
)
rate = 1  # Use 1kHz sampling, like EGI

# first create a new stream info (here we set the name to BioSemi,
# the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
# last value would be the serial number of the device or some other more or
# less locally unique identifier for the stream as far as available (you
# could also omit it but interrupted connections wouldn't auto-recover).
info = StreamInfo(stream_name, "SSVEP_TRIAL_LABEL", number_channels, rate, "float32", "myuid34234")

# next make an outlet
outlet = StreamOutlet(info)

print("now sending data...")
for label in predictions:
    print("Sending " + str(label))
    for i in range(changeRate):
        samples = numpy.zeros(number_channels)
        samples[0] = label
        outlet.push_sample(samples)
        time.sleep(1.0 / rate)
