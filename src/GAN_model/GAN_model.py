import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
from ..G_baseline_batch.G_model import G
from ..D_baseline.D_baseline_model import D

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0


class GAN_model(nn.Module):
    def __init__(self, e_input_size, e_hidden_size, d_input_size, d_hidden_size, output_size):
        self.G = G(e_input_size, e_hidden_size)  # TODO
        self.D = D(d_input_size, d_hidden_size, output_size, self.encoder)  # TODO

    def train(self, **kwargs):
        pass

    def test(self):
        pass

    # L2 loss instead of Binary cross entropy loss (this is optional for stable training)
    def loss(self, D_real, D_fake, gen_params, disc_params, cond_real_data, cond_fake_data, mode, lr=None):
        mode = mode.lower()
        if mode == 'gan':
            G_loss = -torch.mean(self.log(D_fake))
            D_loss = -torch.mean(self.log(1 - D_fake)) - torch.mean(self.log(D_real))
            metric = -D_loss / 2 + np.log(2)  # JS divergence

            G_solver = torch.optim.Adam(gen_params, lr=lr if lr else 1e-3)
            D_solver = torch.optim.Adam(disc_params, lr=lr if lr else 1e-3)

        elif mode == 'lsgan-1':
            G_loss = torch.mean(D_fake ** 2)
            D_loss = torch.mean((D_real - 1) ** 2)
            metric = 0  # TBD

            G_solver = torch.optim.Adam(gen_params, lr=lr if lr else 1e-3)
            D_solver = torch.optim.Adam(disc_params, lr=lr if lr else 1e-3)

        elif mode == 'lsgan-2':
            G_loss = torch.mean((D_fake - 1) ** 2)
            D_loss = torch.mean((D_real - 1) ** 2) + torch.mean(D_fake ** 2)
            metric = D_loss / 2  # Pearson Chi-Square divergence

            G_solver = torch.optim.Adam(gen_params, lr=lr if lr else 1e-3)
            D_solver = torch.optim.Adam(disc_params, lr=lr if lr else 1e-3)

        elif mode == 'wgan':
            G_loss = -torch.mean(D_fake)
            D_loss = torch.mean(D_fake) - torch.mean(D_real)
            metric = -D_loss  # Earth-mover distance

            grad_penalty = self.cal_grad_penalty(cond_real_data, cond_fake_data)
            D_loss += self.lmd * grad_penalty

            G_solver = torch.optim.Adam(gen_params, lr=lr if lr else 1e-3)
            D_solver = torch.optim.Adam(disc_params, lr=lr if lr else 1e-3)

        elif mode == 'bgan':
            G_loss = 0.5 * torch.mean((self.log(D_fake) - self.log(1 - D_fake)) ** 2)
            D_loss = -torch.mean(self.log(D_real) + self.log(1 - D_fake))
            metric = 0  # TBD
            G_solver = torch.optim.Adam(gen_params, lr=lr if lr else 1e-3)
            D_solver = torch.optim.Adam(disc_params, lr=lr if lr else 1e-3)

        else:
            raise ValueError('Unknown mode: {}'.format(mode))

        return G_loss, D_loss, metric, G_solver, D_solver

    def cal_grad_penalty(self, cond_real_data, cond_fake_data):
        epsilon = torch.rand(self.batch_size, 1)
        epsilon = epsilon.expand(cond_real_data.size())

        data_diff = cond_fake_data - cond_real_data
        cond_interp_data = cond_real_data + epsilon * data_diff
        disc_interp = self.D(self.d_net, cond_interp_data, reuse=True)  # TODO: change the arguments

        grad_interp = autograd.grad(outputs=disc_interp, inputs=cond_interp_data,
                                  grad_outputs=torch.ones(disc_interp.size()).cuda(
                                      gpu) if use_cuda else torch.ones(
                                      disc_interp.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        grad_interp_flat = grad_interp.view([self.batch_size, -1])
        slope = grad_interp_flat.norm(p=2, dim=1)

        grad_penalty = torch.mean((slope - 1.) ** 2)
        return grad_penalty
