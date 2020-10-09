# coding=utf-8
from torch import nn
import torch
from genbci.generate.layers.reshape import Reshape, PixelShuffle2d
from genbci.generate.layers.normalization import PixelNorm
from genbci.generate.layers.weight_scaling import weight_scale
from genbci.generate.layers.upsampling import CubicUpsampling1d, CubicUpsampling2d
from genbci.generate.layers.stdmap import StdMap1d
from genbci.generate.progressive import (
    ProgressiveGenerator,
    ProgressiveGeneratorBlock,
    ProgressiveDiscriminator,
    ProgressiveDiscriminatorBlock,
)
from genbci.generate.wgan import WGAN_I_Generator, WGAN_I_Discriminator
from genbci.generate.wgan import WGAN_Generator, WGAN_Discriminator
from genbci.generate.wgan import WGAN_I_CDiscriminator, WGAN_I_CGenerator
from torch.nn.init import calculate_gain


def create_disc_blocks(n_chans):
    def create_conv_sequence(in_filters, out_filters):
        return nn.Sequential(
            weight_scale(
                nn.Conv1d(in_filters, in_filters, 9, padding=4),
                gain=calculate_gain("leaky_relu"),
            ),
            nn.LeakyReLU(0.2),
            weight_scale(
                nn.Conv1d(in_filters, out_filters, 9, padding=4),
                gain=calculate_gain("leaky_relu"),
            ),
            nn.LeakyReLU(0.2),
            weight_scale(
                nn.Conv1d(out_filters, out_filters, 2, stride=2),
                gain=calculate_gain("leaky_relu"),
            ),
            nn.LeakyReLU(0.2),
        )

    def create_in_sequence(n_chans, out_filters):
        return nn.Sequential(
            weight_scale(
                nn.Conv2d(1, out_filters, (1, n_chans)),
                gain=calculate_gain("leaky_relu"),
            ),
            Reshape([[0], [1], [2]]),
            nn.LeakyReLU(0.2),
        )

    def create_fade_sequence(factor):
        return nn.AvgPool2d((factor, 1), stride=(factor, 1))

    blocks = []
    tmp_block = ProgressiveDiscriminatorBlock(
        create_conv_sequence(50, 50),
        create_in_sequence(n_chans, 50),
        create_fade_sequence(2),
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveDiscriminatorBlock(
        create_conv_sequence(50, 50),
        create_in_sequence(n_chans, 50),
        create_fade_sequence(2),
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveDiscriminatorBlock(
        create_conv_sequence(50, 50),
        create_in_sequence(n_chans, 50),
        create_fade_sequence(2),
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveDiscriminatorBlock(
        create_conv_sequence(50, 50),
        create_in_sequence(n_chans, 50),
        create_fade_sequence(2),
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveDiscriminatorBlock(
        create_conv_sequence(50, 50),
        create_in_sequence(n_chans, 50),
        create_fade_sequence(2),
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveDiscriminatorBlock(
        nn.Sequential(
            StdMap1d(),
            create_conv_sequence(51, 50),
            Reshape([[0], -1]),
            weight_scale(nn.Linear(50 * 27, 1), gain=calculate_gain("linear")),
        ),
        create_in_sequence(n_chans, 50),
        None,
    )
    blocks.append(tmp_block)
    return blocks


def create_gen_blocks(n_chans, z_vars):
    def create_conv_sequence(in_filters, out_filters):
        return nn.Sequential(
            nn.Upsample(mode="linear", scale_factor=2),
            weight_scale(
                nn.Conv1d(in_filters, out_filters, 9, padding=4),
                gain=calculate_gain("leaky_relu"),
            ),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            weight_scale(
                nn.Conv1d(out_filters, out_filters, 9, padding=4),
                gain=calculate_gain("leaky_relu"),
            ),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

    def create_out_sequence(n_chans, in_filters):
        return nn.Sequential(
            weight_scale(
                nn.Conv1d(in_filters, n_chans, 1), gain=calculate_gain("linear")
            ),
            Reshape([[0], [1], [2], 1]),
            PixelShuffle2d([1, n_chans]),
        )

    def create_fade_sequence(factor):
        return nn.Upsample(mode="bilinear", scale_factor=(2, 1))

    blocks = []
    tmp_block = ProgressiveGeneratorBlock(
        nn.Sequential(
            weight_scale(nn.Linear(z_vars, 50 * 27), gain=calculate_gain("leaky_relu")),
            nn.LeakyReLU(0.2),
            Reshape([[0], 50, -1]),
            create_conv_sequence(50, 50),
        ),
        create_out_sequence(n_chans, 50),
        create_fade_sequence(2),
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveGeneratorBlock(
        create_conv_sequence(50, 50),
        create_out_sequence(n_chans, 50),
        create_fade_sequence(2),
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveGeneratorBlock(
        create_conv_sequence(50, 50),
        create_out_sequence(n_chans, 50),
        create_fade_sequence(2),
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveGeneratorBlock(
        create_conv_sequence(50, 50),
        create_out_sequence(n_chans, 50),
        create_fade_sequence(2),
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveGeneratorBlock(
        create_conv_sequence(50, 50),
        create_out_sequence(n_chans, 50),
        create_fade_sequence(2),
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveGeneratorBlock(
        create_conv_sequence(50, 50), create_out_sequence(n_chans, 50), None
    )
    blocks.append(tmp_block)
    return blocks


### PROGRESSIVE


class Generator(WGAN_I_Generator):
    def __init__(self, n_chans, z_vars):
        super(Generator, self).__init__()
        self.model = ProgressiveGenerator(create_gen_blocks(n_chans, z_vars))

    def forward(self, input):
        return self.model(input)


class Discriminator(WGAN_I_Discriminator):
    def __init__(self, n_chans):
        super(Discriminator, self).__init__()
        self.model = ProgressiveDiscriminator(create_disc_blocks(n_chans))

    def forward(self, input):
        return self.model(input)


### PLAIN
class SSVEP_Generator(WGAN_Generator):
    def __init__(self, nz):
        super(SSVEP_Generator, self).__init__()
        self.nz = nz
        self.layer1 = nn.Sequential(nn.Linear(self.nz, 560), nn.PReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=16, out_channels=16, kernel_size=18, stride=2
            ),
            nn.PReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=16, out_channels=16, kernel_size=18, stride=4
            ),
            nn.PReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=16, out_channels=2, kernel_size=14, stride=2
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        out = self.layer1(input)
        out = out.view(out.size(0), 16, 35)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class SSVEP_Discriminator(WGAN_Discriminator):
    def __init__(self):
        super(SSVEP_Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=10, stride=2),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(2880, 600),
            nn.LeakyReLU(0.2),
            nn.Linear(600, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, input):

        out = self.layer1(input)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out


### IMPROVED


class SSVEP_Generator_I(WGAN_I_Generator):
    def __init__(self, nz):
        super(SSVEP_Generator_I, self).__init__()
        self.nz = nz
        self.layer1 = nn.Sequential(nn.Linear(self.nz, 560), nn.PReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=16, out_channels=16, kernel_size=18, stride=2
            ),
            nn.PReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=16, out_channels=16, kernel_size=18, stride=4
            ),
            nn.PReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=16, out_channels=2, kernel_size=14, stride=2
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        out = self.layer1(input)
        out = out.view(out.size(0), 16, 35)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class SSVEP_Discriminator_I(WGAN_I_Discriminator):
    def __init__(self):
        super(SSVEP_Discriminator_I, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=10, stride=2),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(2880, 600),
            nn.LeakyReLU(0.2),
            nn.Linear(600, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, input):

        out = self.layer1(input)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out


### Conditional
class SSVEP_CGenerator(WGAN_I_CGenerator):
    def __init__(self, nz, nclasses):
        super(SSVEP_CGenerator, self).__init__()

        self.label_emb = nn.Embedding(nclasses, nclasses)
        self.nz = nz
        self.layer1 = nn.Sequential(nn.Linear(self.nz + nclasses, 640), nn.PReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=16, out_channels=16, kernel_size=22, stride=4
            ),
            nn.PReLU(),
        )
        # self.layer3 = nn.Sequential(
        #     nn.ConvTranspose1d(
        #         in_channels=16, out_channels=16, kernel_size=18, stride=2
        #     ),
        #     nn.PReLU(),
        # )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=16, out_channels=2, kernel_size=20, stride=4
            ),
            nn.Sigmoid(),
        )

    def forward(self, input, labels):

        gen_input = torch.cat((self.label_emb(labels), input), -1)
        out = self.layer1(gen_input)
        out = out.view(out.size(0), 16, 40)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        return out


class SSVEP_CDiscriminator(WGAN_I_CDiscriminator):
    def __init__(self, nclasses):
        super(SSVEP_CDiscriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=10, stride=2),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
        )

        self.fc_labels = nn.Sequential(nn.Linear(1, 1000), nn.LeakyReLU(0.2))
        self.dense_layers = nn.Sequential(
            nn.Linear(2880 + 1000, 600), nn.LeakyReLU(0.2), nn.Linear(600, 1)
        )

    def forward(self, input, labels):
        out = self.layer1(input)
        out = out.view(out.size(0), -1)

        out_labels = self.fc_labels(labels.float().unsqueeze(-1))

        out = torch.cat([out, out_labels], -1)
        out = self.dense_layers(out)
        return out
