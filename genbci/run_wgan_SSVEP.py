import argparse
import numpy as np
import random
import os

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from genbci.generate.wgan import WGAN_Discriminator, WGAN_Generator
from genbci.scripts import dataprep_ssvep


def init_torch_and_get_device(random_state=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    random.seed(random_state)

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--modelpath", type=str, default="models/", help="Path to dave model"
)
parser.add_argument(
    "--n_epochs", type=int, default=1200, help="number of epochs of training"
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
opt.batch_size = 5
opt.dropout_level = 0.05
opt.img_shape = (9, 1500)
opt.T = 3.0


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


for nclass in range(0, 3):

    if nclass == 0:
        data_train = dataprep_ssvep.data_class0
    if nclass == 1:
        data_train = dataprep_ssvep.data_class1
        label_train = dataprep_ssvep.label_class1
    if nclass == 2:
        data_train = dataprep_ssvep.data_class2
        label_train = dataprep_ssvep.label_class2

    data_train = data_train.swapaxes(1, 2)
print(data_train.shape)
datatrain = torch.from_numpy(data_train)
label = torch.from_numpy(label_train)

dataset = torch.utils.data.TensorDataset(datatrain, label)
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

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise for generator input
            z = torch.randn(opt.batch_size, opt.nz).to(opt.device)

            # Generate a batch of fake images
            fake_imgs = generator(z)

            # Let the discriminator judge and learn
            loss_real_d, loss_fake_d = discriminator.train_batch(real_imgs, fake_imgs)
            loss_d = loss_real_d + loss_fake_d
            losses_d.append(loss_d)

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                z = torch.randn(opt.batch_size, opt.nz).to(opt.device)
                loss_g = generator.train_batch(z, discriminator)
                losses_g.append(loss_g)

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), loss_d, loss_g)
                )

                eval_fn(dataloader, generator, discriminator, epoch, opt)


def eval_fn(dataloader, generator, discriminator, epoch, opt):

    generator.eval()
    discriminator.eval()

    # TODO Implement checkpointing
    # TODO Fix ugly indexing
    freqs_tmp = np.fft.rfftfreq(data_train.shape[2], d=1 / 250.0)

    # Compute FFT frequencies
    train_fft = np.fft.rfft(data_train, axis=2)

    # Compute FFT on training data
    train_amps = np.abs(train_fft).mean(axis=1).mean(axis=0)

    # Noise for generator
    z = torch.rand(opt.batch_size, opt.nz).to(opt.device)

    # Get a batch of fake data and compute FFT
    batch_fake = generator(z)
    fake_fft = np.fft.rfft(batch_fake.data.cpu().numpy(), axis=2)
    fake_amps = np.abs(fake_fft).mean(axis=1).mean(axis=0)

    plt.figure()
    plt.plot(freqs_tmp, np.log(fake_amps), label="Fake")
    plt.plot(freqs_tmp, np.log(train_amps), label="Real")
    plt.title("Frequency Spectrum")
    plt.xlabel("Hz")
    plt.legend()
    plt.savefig(os.path.join(opt.modelpath, "_fft_%d.png" % epoch))
    plt.close()


class Generator(WGAN_Generator):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.layer1 = nn.Sequential(nn.Linear(self.nz, 640), nn.PReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=16, out_channels=16, kernel_size=22, stride=4
            ),
            nn.PReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=16, out_channels=16, kernel_size=18, stride=2
            ),
            nn.PReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=16, out_channels=2, kernel_size=16, stride=4
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        out = self.layer1(input)
        out = out.view(out.size(0), 16, 40)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class Discriminator(WGAN_Discriminator):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=10, stride=2),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(5968, 600), nn.LeakyReLU(0.2), nn.Linear(600, 1)
        )

    def forward(self, input):
        out = self.layer1(input)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out


# Initialize generator and discriminator
discriminator = Discriminator()
discriminator.apply(weights_init)
discriminator.train_init()
discriminator.to(opt.device)


generator = Generator(opt.nz)
generator.apply(weights_init)
generator.train_init()
generator.to(opt.device)


train_fn(dataloader, generator, discriminator, opt)
