# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from genbci.util import *
from genbci.generate.gan import GAN_Discriminator
from genbci.generate.gan import GAN_Generator


class WGAN_Discriminator(GAN_Discriminator):
    """
    Wasserstein GAN discriminator
    References
    ----------
    Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN.
    Retrieved from http://arxiv.org/abs/1701.07875
    """

    def __init__(self):
        super(WGAN_Discriminator, self).__init__()

    def train_init(self, lr=1e-4, c=0.01):
        """
        Initialize RMS optimizer and weight clipping for discriminator
        Parameters
        ----------
        alpha : float, optional
            Learning rate for Adam
        c : float, optional
            Limits for weight clipping
        """
        self.c = c
        for p in self.parameters():
            p.data.clamp_(-self.c, self.c)

        self.loss = None
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.did_init_train = True

    def update_parameters(self):
        super(WGAN_Discriminator, self).update_parameters()
        for p in self.parameters():
            p.data.clamp_(-self.c, self.c)

    def train_batch(self, batch_real, batch_fake):
        """
        Train discriminator for one batch of real and fake data
        Parameters
        ----------
        batch_real : autograd.Variable
            Batch of real data
        batch_fake : autograd.Variable
            Batch of fake data
        Returns
        -------
        loss_real : float
            WGAN loss for real data
        loss_fake : float
            WGAN loss for fake data
        """
        self.pre_train()

        # Compute output and loss
        fx_real = self.forward(batch_real)
        loss_real = -torch.mean(fx_real)
        loss_real.backward()

        fx_fake = self.forward(batch_fake)
        loss_fake = torch.mean(fx_fake)
        loss_fake.backward()

        loss = loss_real + loss_fake

        self.update_parameters()

        loss_real = loss_real.item()
        loss_fake = loss_fake.item()
        return loss_real, loss_fake  # return loss


class WGAN_Generator(GAN_Generator):
    """
    Wasserstein GAN generator
    References
    ----------
    Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN.
    Retrieved from http://arxiv.org/abs/1701.07875
    """

    def __init__(self):
        super(WGAN_Generator, self).__init__()

    def train_init(self, lr=1e-4):
        """
        Initialize RMS optimizer for generator
        Parameters
        ----------
        alpha : float, optional
            Learning rate for Adam
        """
        self.loss = None
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.did_init_train = True

    def train_batch(self, batch_noise, discriminator):
        """
        Train generator for one batch of latent noise
        Parameters
        ----------
        batch_noise : autograd.Variable
            Batch of latent noise
        discriminator : nn.Module
            Discriminator to evaluate realness of generated data
        Returns
        -------
        loss : float
            WGAN loss against evaluation of discriminator of generated samples
            to be real
        """
        self.pre_train(discriminator)

        # Generate and discriminate
        gen = self.forward(batch_noise)
        disc = discriminator(gen)
        loss = -torch.mean(disc)

        # Backprop gradient
        loss.backward()

        # Update parameters
        self.update_parameters()

        loss = loss.item()
        return loss  # return loss


class WGAN_I_Discriminator(GAN_Discriminator):
    """
    Improved Wasserstein GAN discriminator
    References
    ----------
    Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).
    Improved Training of Wasserstein GANs.
    Retrieved from http://arxiv.org/abs/1704.00028
    """

    def __init__(self):
        super(WGAN_I_Discriminator, self).__init__()

    def train_init(
        self,
        alpha=1e-4,
        betas=(0.5, 0.9),
        lambd=10,
        one_sided_penalty=False,
        distance_weighting=False,
        eps_drift=0.0,
        eps_center=0.0,
        lambd_consistency_term=0.0,
    ):
        """
        Initialize Adam optimizer for discriminator
        Parameters
        ----------
        alpha : float, optional
            Learning rate for Adam
        betas : (float,float), optional
            Betas for Adam
        lambda : float, optional
            Weight for gradient penalty (default: 10)
        one_sided_penalty : bool, optional
            Use one- or two-sided penalty
            See Hartmann et al., 2018 (default: False)
        distance_weighting : bool
            Use distance-weighting
            See Hartmann et al., 2018 (default: False)
        eps_drift : float, optional
            Weigth to keep discriminator output from drifting away from 0
            See Karras et al., 2017 (default: 0.)
        eps_center : float, optional
            Weight to keep discriminator centered at 0
            See Hartmann et al., 2018 (default: 0.)
        References
        ----------
        Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
        Progressive Growing of GANs for Improved Quality, Stability,
        and Variation. Retrieved from http://arxiv.org/abs/1710.10196
        Hartmann, K. G., Schirrmeister, R. T., & Ball, T. (2018).
        EEG-GAN: Generative adversarial networks for electroencephalograhic
        (EEG) brain signals. Retrieved from https://arxiv.org/abs/1806.01875
        """
        super(WGAN_I_Discriminator, self).train_init(alpha, betas)
        self.loss = None
        self.lambd = lambd
        self.one_sided_penalty = one_sided_penalty
        self.distance_weighting = distance_weighting
        self.eps_drift = eps_drift
        self.eps_center = eps_center
        self.lambd_consistency_term = lambd_consistency_term

    def train_batch(self, batch_real, batch_fake):
        """
        Train discriminator for one batch of real and fake data
        Parameters
        ----------
        batch_real : autograd.Variable
            Batch of real data
        batch_fake : autograd.Variable
            Batch of fake data
        Returns
        -------
        loss_real : float
            WGAN loss for real data
        loss_fake : float
            WGAN loss for fake data
        loss_penalty : float
            Improved WGAN penalty term
        loss_drift : float
            Drifting penalty
        loss_center : float
            Center penalty
        """
        self.pre_train()

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        batch_real, one, mone = cuda_check([batch_real, one, mone])

        fx_real = self(batch_real)
        loss_real = fx_real.mean()
        loss_real.backward(
            mone, retain_graph=(self.eps_drift > 0 or self.eps_center > 0)
        )

        fx_fake = self(batch_fake)
        loss_fake = fx_fake.mean()
        loss_fake.backward(
            one, retain_graph=(self.eps_drift > 0 or self.eps_center > 0)
        )

        loss_drift = 0
        loss_center = 0
        if self.eps_drift > 0:
            tmp_drift = self.eps_drift * loss_real ** 2
            tmp_drift.backward(retain_graph=self.eps_center > 0)
            loss_drift = tmp_drift.item()
        if self.eps_center > 0:
            tmp_center = loss_real + loss_fake
            tmp_center = self.eps_center * tmp_center ** 2
            tmp_center.backward()
            loss_center = tmp_center.item()

        # loss_consistency_term
        # if self.lambd_consistency_term>0:
        #    batch_real_1

        dist = 1
        if self.distance_weighting:
            dist = (loss_real - loss_fake).detach()
            dist = dist.clamp(min=0)
        loss_penalty = self.calc_gradient_penalty(batch_real, batch_fake)
        loss_penalty = self.lambd * dist * loss_penalty
        loss_penalty.backward()

        # Update parameters
        self.update_parameters()

        loss_real = -loss_real.item()
        loss_fake = loss_fake.item()
        loss_penalty = loss_penalty.item()
        return (
            loss_real,
            loss_fake,
            loss_penalty,
            loss_drift,
            loss_center,
        )  # return loss

    def calc_gradient_penalty(self, batch_real, batch_fake):
        """
        Improved WGAN gradient penalty
        Parameters
        ----------
        batch_real : autograd.Variable
            Batch of real data
        batch_fake : autograd.Variable
            Batch of fake data
        Returns
        -------
        gradient_penalty : autograd.Variable
            Gradient penalties
        """
        alpha = torch.rand(
            batch_real.data.size(0), *((len(batch_real.data.size()) - 1) * [1])
        )
        alpha = alpha.expand(batch_real.data.size())
        batch_real, alpha = cuda_check([batch_real, alpha])

        interpolates = alpha * batch_real.data + ((1 - alpha) * batch_fake.data)
        interpolates = Variable(interpolates, requires_grad=True)
        alpha, interpolates = cuda_check([alpha, interpolates])

        disc_interpolates = self(interpolates)

        ones = torch.ones(disc_interpolates.size())
        interpolates, ones = cuda_check([interpolates, ones])

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        tmp = gradients.norm(2, dim=1) - 1
        if self.one_sided_penalty:
            tmp = tmp.clamp(min=0)
        gradient_penalty = ((tmp) ** 2).mean()

        return gradient_penalty


class WGAN_I_Generator(GAN_Generator):
    """
    Improved Wasserstein GAN generator
    References
    ----------
    Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).
    Improved Training of Wasserstein GANs.
    Retrieved from http://arxiv.org/abs/1704.00028
    """

    def __init__(self):
        super(WGAN_I_Generator, self).__init__()

    def train_init(self, alpha=1e-4, betas=(0.5, 0.9)):
        """
        Initialize Adam optimizer for generator
        Parameters
        ----------
        alpha : float, optional
            Learning rate for Adam
        betas : (float,float), optional
            Betas for Adam
        """
        self.loss = None
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, betas=betas)
        self.did_init_train = True

    def train_batch(self, batch_noise, discriminator):
        """
        Train generator for one batch of latent noise
        Parameters
        ----------
        batch_noise : autograd.Variable
            Batch of latent noise
        discriminator : nn.Module
            Discriminator to evaluate realness of generated data
        Returns
        -------
        loss : float
            WGAN loss against evaluation of discriminator of generated samples
            to be real
        """
        self.pre_train(discriminator)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        batch_noise, mone = cuda_check([batch_noise, mone])

        # Generate and discriminate
        gen = self(batch_noise)
        disc = discriminator(gen)
        loss = disc.mean()
        # Backprop gradient
        loss.backward(mone)

        # Update parameters
        self.update_parameters()

        loss = loss.item()
        return loss  # return loss


class WGAN_I_CDiscriminator(GAN_Discriminator):
    """
    Improved Wasserstein GAN discriminator
    References
    ----------
    Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).
    Improved Training of Wasserstein GANs.
    Retrieved from http://arxiv.org/abs/1704.00028
    """

    def __init__(self):
        super(WGAN_I_CDiscriminator, self).__init__()

    def train_init(
        self,
        alpha=1e-4,
        betas=(0.5, 0.9),
        lambd=10,
        one_sided_penalty=False,
        distance_weighting=False,
        eps_drift=0.0,
        eps_center=0.0,
        lambd_consistency_term=0.0,
    ):
        """
        Initialize Adam optimizer for discriminator
        Parameters
        ----------
        alpha : float, optional
            Learning rate for Adam
        betas : (float,float), optional
            Betas for Adam
        lambda : float, optional
            Weight for gradient penalty (default: 10)
        one_sided_penalty : bool, optional
            Use one- or two-sided penalty
            See Hartmann et al., 2018 (default: False)
        distance_weighting : bool
            Use distance-weighting
            See Hartmann et al., 2018 (default: False)
        eps_drift : float, optional
            Weigth to keep discriminator output from drifting away from 0
            See Karras et al., 2017 (default: 0.)
        eps_center : float, optional
            Weight to keep discriminator centered at 0
            See Hartmann et al., 2018 (default: 0.)
        References
        ----------
        Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
        Progressive Growing of GANs for Improved Quality, Stability,
        and Variation. Retrieved from http://arxiv.org/abs/1710.10196
        Hartmann, K. G., Schirrmeister, R. T., & Ball, T. (2018).
        EEG-GAN: Generative adversarial networks for electroencephalograhic
        (EEG) brain signals. Retrieved from https://arxiv.org/abs/1806.01875
        """
        super(WGAN_I_CDiscriminator, self).train_init(alpha, betas)
        self.loss = None
        self.lambd = lambd
        self.one_sided_penalty = one_sided_penalty
        self.distance_weighting = distance_weighting
        self.eps_drift = eps_drift
        self.eps_center = eps_center
        self.lambd_consistency_term = lambd_consistency_term

    def train_batch(self, batch_real, batch_fake, labels):
        """
        Train discriminator for one batch of real and fake data
        Parameters
        ----------
        batch_real : autograd.Variable
            Batch of real data
        batch_fake : autograd.Variable
            Batch of fake data
        Returns
        -------
        loss_real : float
            WGAN loss for real data
        loss_fake : float
            WGAN loss for fake data
        loss_penalty : float
            Improved WGAN penalty term
        loss_drift : float
            Drifting penalty
        loss_center : float
            Center penalty
        """
        self.pre_train()

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        batch_real, one, mone = cuda_check([batch_real, one, mone])

        fx_real = self(batch_real, labels)
        loss_real = fx_real.mean()
        loss_real.backward(
            mone, retain_graph=(self.eps_drift > 0 or self.eps_center > 0)
        )

        fx_fake = self(batch_fake, labels)
        loss_fake = fx_fake.mean()
        loss_fake.backward(
            one, retain_graph=(self.eps_drift > 0 or self.eps_center > 0)
        )

        loss_drift = 0
        loss_center = 0
        if self.eps_drift > 0:
            tmp_drift = self.eps_drift * loss_real ** 2
            tmp_drift.backward(retain_graph=self.eps_center > 0)
            loss_drift = tmp_drift.item()
        if self.eps_center > 0:
            tmp_center = loss_real + loss_fake
            tmp_center = self.eps_center * tmp_center ** 2
            tmp_center.backward()
            loss_center = tmp_center.item()

        # loss_consistency_term
        # if self.lambd_consistency_term>0:
        #    batch_real_1

        dist = 1
        if self.distance_weighting:
            dist = (loss_real - loss_fake).detach()
            dist = dist.clamp(min=0)
        loss_penalty = self.calc_gradient_penalty(batch_real, batch_fake, labels)
        loss_penalty = self.lambd * dist * loss_penalty
        loss_penalty.backward()

        # Update parameters
        self.update_parameters()

        loss_real = -loss_real.item()
        loss_fake = loss_fake.item()
        loss_penalty = loss_penalty.item()
        return (
            loss_real,
            loss_fake,
            loss_penalty,
            loss_drift,
            loss_center,
        )  # return loss

    def calc_gradient_penalty(self, batch_real, batch_fake, labels):
        """
        Improved WGAN gradient penalty
        Parameters
        ----------
        batch_real : autograd.Variable
            Batch of real data
        batch_fake : autograd.Variable
            Batch of fake data
        Returns
        -------
        gradient_penalty : autograd.Variable
            Gradient penalties
        """
        alpha = torch.rand(
            batch_real.data.size(0), *((len(batch_real.data.size()) - 1) * [1])
        )
        alpha = alpha.expand(batch_real.data.size())
        batch_real, alpha = cuda_check([batch_real, alpha])

        interpolates = alpha * batch_real.data + ((1 - alpha) * batch_fake.data)
        interpolates = Variable(interpolates, requires_grad=True)
        alpha, interpolates = cuda_check([alpha, interpolates])

        disc_interpolates = self(interpolates, labels)

        ones = torch.ones(disc_interpolates.size())
        interpolates, ones = cuda_check([interpolates, ones])

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        tmp = gradients.norm(2, dim=1) - 1
        if self.one_sided_penalty:
            tmp = tmp.clamp(min=0)
        gradient_penalty = ((tmp) ** 2).mean()

        return gradient_penalty


class WGAN_I_CGenerator(GAN_Generator):
    """
    Improved Wasserstein GAN generator
    References
    ----------
    Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).
    Improved Training of Wasserstein GANs.
    Retrieved from http://arxiv.org/abs/1704.00028
    """

    def __init__(self):
        super(WGAN_I_CGenerator, self).__init__()

    def train_init(self, alpha=1e-4, betas=(0.5, 0.9)):
        """
        Initialize Adam optimizer for generator
        Parameters
        ----------
        alpha : float, optional
            Learning rate for Adam
        betas : (float,float), optional
            Betas for Adam
        """
        self.loss = None
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, betas=betas)
        self.did_init_train = True

    def train_batch(self, batch_noise, discriminator, labels):
        """
        Train generator for one batch of latent noise
        Parameters
        ----------
        batch_noise : autograd.Variable
            Batch of latent noise
        discriminator : nn.Module
            Discriminator to evaluate realness of generated data
        Returns
        -------
        loss : float
            WGAN loss against evaluation of discriminator of generated samples
            to be real
        """
        self.pre_train(discriminator)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        batch_noise, mone = cuda_check([batch_noise, mone])

        # Generate and discriminate
        gen = self(batch_noise, labels)
        disc = discriminator(gen, labels)
        loss = disc.mean()
        # Backprop gradient
        loss.backward(mone)

        # Update parameters
        self.update_parameters()

        loss = loss.item()
        return loss  # return loss

