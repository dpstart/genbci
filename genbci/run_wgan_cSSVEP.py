import argparse
import numpy as np
import random
import os

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from genbci.generate.model import (
    SSVEP_CDiscriminator as Discriminator,
    SSVEP_CGenerator as Generator,
)

# from genbci.scripts import ssvep_sample
from genbci.util import init_torch_and_get_device, weights_init, get_exo_data

torch.set_num_threads(8)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--modelpath", type=str, default="models/", help="Path to dave model"
)
parser.add_argument(
    "--n_epochs", type=int, default=5000, help="number of epochs of training"
)
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument(
    "--b1",
    type=float,
    default=0.1,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=4,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--latent_dim", type=int, default=32, help="dimensionality of the latent space"
)
parser.add_argument(
    "--img_size", type=int, default=28, help="size of each image dimension"
)
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument(
    "--n_critic",
    type=int,
    default=5,
    help="number of training steps for discriminator per iter",
)
parser.add_argument(
    "--clip_value",
    type=float,
    default=0.01,
    help="lower and upper clip value for disc. weights",
)
parser.add_argument(
    "--sample_interval", type=int, default=200, help="interval between image samples"
)
parser.add_argument(
    "--nz",
    type=int,
    default=64,
    help="size of the latent z vector used as the generator input.",
)
opt = parser.parse_args()
opt.device = init_torch_and_get_device()

### Setting some defaults
opt.batch_size = 16
opt.dropout_level = 0.05
# opt.img_shape = (9, 1500)
opt.plot_steps = 250

opt.jobid = 2

opt.modelname = "ssvep_wgan%s"
if not os.path.exists(opt.modelpath):
    os.makedirs(opt.modelpath)

# dataloader = torch.utils.data.DataLoader(
#     dataset=ssvep_sample.dataset, batch_size=opt.batch_size, shuffle=True
# )


epochs_exo = get_exo_data(
    "/Users/daniele/Desktop/thesis/library/genbci/ssvep/data/dataset-ssvep-exoskeleton",
    plot=False,
)

data = epochs_exo.get_data()
labels = epochs_exo.events[:, 2] - 1

# Electrodes 2 and 3 should be O1 and O2 thus occipital
datatrain = torch.from_numpy(data[:, 1:3, :728]).float()
labels = torch.from_numpy(labels)
dataset = torch.utils.data.TensorDataset(datatrain, labels)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=opt.batch_size, shuffle=True
)


def train_fn(dataloader, generator, discriminator, opt):

    losses_d, losses_g = [], []

    for epoch in range(opt.n_epochs):
        for i, (real_imgs, labels) in enumerate(dataloader):

            generator.train()
            discriminator.train()

            real_imgs = real_imgs.to(opt.device)
            labels = labels.long().to(opt.device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise for generator input
            z = torch.randn(opt.batch_size, opt.nz).to(opt.device)

            # Generate a batch of fake images
            fake_imgs = generator(z, labels)

            # Let the discriminator judge and learn
            (loss_real_d, loss_fake_d, _, _, _) = discriminator.train_batch(
                real_imgs, fake_imgs, labels
            )
            loss_d = loss_real_d + loss_fake_d
            losses_d.append(loss_d)

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                z = torch.randn(opt.batch_size, opt.nz).to(opt.device)
                loss_g = generator.train_batch(z, discriminator, labels)
                losses_g.append(loss_g)

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), loss_d, loss_g)
                )

                eval_fn(
                    dataloader,
                    generator,
                    discriminator,
                    epoch,
                    opt,
                    losses_d,
                    losses_g,
                    labels,
                )


def eval_fn(
    dataloader, generator, discriminator, epoch, opt, losses_d, losses_g, labels
):

    generator.eval()
    discriminator.eval()

    if epoch % opt.plot_steps == 0:

        freqs_tmp = np.fft.rfftfreq(dataset.tensors[0].shape[2], d=1 / 250.0)

        # Compute FFT frequencies
        train_fft = np.fft.rfft(dataset.tensors[0], axis=2)

        # Compute FFT on training data
        train_amps = np.abs(train_fft).mean(axis=1).mean(axis=0)

        # Noise for generator
        z = torch.rand(opt.batch_size, opt.nz).to(opt.device)

        # Get a batch of fake data and compute FFT
        batch_fake = generator(z, labels)
        fake_fft = np.fft.rfft(batch_fake.data.cpu().numpy(), axis=2)
        fake_amps = np.abs(fake_fft).mean(axis=1).mean(axis=0)

        plt.figure()
        plt.plot(freqs_tmp, np.log(fake_amps), label="Fake")
        plt.plot(freqs_tmp, np.log(train_amps), label="Real")
        plt.title("Frequency Spectrum")
        plt.xlabel("Hz")
        plt.legend()
        plt.savefig(
            os.path.join(
                opt.modelpath, opt.modelname % opt.jobid + "_fft_%d.png" % epoch
            )
        )
        plt.close()

        batch_fake = batch_fake.data.cpu().numpy()

        plt.figure(figsize=(10, 10))
        for i in range(10):
            plt.subplot(10, 1, i + 1)

            # Working with 2 channels, plot only firt one. A bit ugly.
            plt.plot(batch_fake[i, 0, ...].squeeze())
            plt.xticks((), ())
            plt.yticks((), ())
        plt.subplots_adjust(hspace=0)
        plt.savefig(
            os.path.join(
                opt.modelpath, opt.modelname % opt.jobid + "_fakes_%d.png" % epoch
            )
        )
        plt.close()

        plt.figure(figsize=(10, 15))
        plt.plot(np.asarray(losses_d))
        plt.title("Loss Discriminator")
        plt.savefig(
            os.path.join(
                opt.modelpath, opt.modelname % opt.jobid + "loss_disc_%d.png" % epoch
            )
        )
        plt.close()

        plt.figure(figsize=(10, 15))
        plt.plot(np.asarray(losses_g))
        plt.title("Loss generator")
        plt.savefig(
            os.path.join(
                opt.modelpath, opt.modelname % opt.jobid + "loss_gen_%d.png" % epoch
            )
        )
        plt.close()

        discriminator.save_model(
            os.path.join(opt.modelpath, opt.modelname % opt.jobid + ".disc")
        )
        generator.save_model(
            os.path.join(opt.modelpath, opt.modelname % opt.jobid + ".gen")
        )


# Initialize generator and discriminator
discriminator = Discriminator(4)
discriminator.apply(weights_init)
discriminator.train_init()
discriminator.to(opt.device)

generator = Generator(opt.nz, 4)
generator.apply(weights_init)
generator.train_init()
generator.to(opt.device)


train_fn(dataloader, generator, discriminator, opt)
