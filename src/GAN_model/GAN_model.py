import sys, os
sys.path.append(os.path.abspath(__file__ + "/../../") + '/G_baseline')
sys.path.append(os.path.abspath(__file__ + "/../../") + '/D_baseline')
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
from data_proc import *
from G_model import *
from D_model import *

import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable

######### helper functions for time recording & logging ##########
import time
import math

# FIXME: added these two functions because import util does not seem to work (see above)
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
###################################################################

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

def to_var(x):
    if use_cuda:
        x = x.cuda()
    return Variable(x)


class GAN_model(nn.Module):
    def __init__(self, G_enc_input_size, G_enc_hidden_size, G_enc_n_layers, G_enc_num_directions,
                 G_dec_input_size, G_dec_hidden_size, G_output_size, G_dec_n_layers, G_dec_num_directions,
                 D_enc_input_size, D_enc_hidden_size, D_enc_n_layers, D_num_directions,
                 D_mlp_hidden_size, D_num_attn_weights, D_mlp_output_size,
                 use_attn, batch_size
                 ):

        super(GAN_model, self).__init__()

        self.G = G(G_enc_input_size, G_enc_hidden_size, G_enc_n_layers, G_enc_num_directions,G_dec_input_size,
                   G_dec_hidden_size, G_output_size, G_dec_n_layers, G_dec_num_directions, batch_size)

        self.D = D(D_enc_input_size, D_enc_hidden_size, D_enc_n_layers, D_num_directions,D_mlp_hidden_size,
                   D_num_attn_weights, D_mlp_output_size, use_attn, batch_size)

    def train(self, triplets, n_iters, d_steps, d_optimizer, g_steps, g_optimizer, batch_size,
              criterion, word2index, embeddings_index, embeddings_size):
        # criterion is for both G and D

        # record start time for logging
        begin_time = time.time()

        for iter in range(1, n_iters + 1):

            # load a minibatch of data from corpus + data from the generator

            # train D
            for d_train_idx in range(d_steps):
                # 1. Train D on real+fake
                self.D.zero_grad()

                #  1A: Train D on real
                #       get data
                #       prepare batch
                training_batch, seq_lens = get_random_batch(triplets, batch_size)
                #       concat the context_ans batch with the question batch
                #       each element in the training batch is context + question + answer
                cqa_batch, _, cqa_lens = prepare_batch_var(training_batch, seq_lens,
                                                                batch_size, word2index, embeddings_index,
                                                                embeddings_size, mode=['word'], concat_opt='cqa')
                ca_batch, _, ca_lens = prepare_batch_var(training_batch, seq_lens,
                                                                batch_size, word2index, embeddings_index,
                                                                embeddings_size, mode=['word'], concat_opt='ca')

                train_input = Variable(cqa_batch[0].cuda()) if use_cuda else Variable(
                    cqa_batch[0])  # embeddings vectors, size = [seq len x batch size x embedding dim]

                d_real_decision = self.D.forward(train_input, cqa_lens[0])
                d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
                d_real_error.backward()  # compute/store gradients, but don't change params

                #  1B: Train D on fake
                d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size)) # TODO replace this line by sampling from generator
                d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
                d_fake_decision = D(preprocess(d_fake_data.t()))
                d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
                # d_fake_error.backward()
                # d_optimizer.step()

                # accumulate loss
                # FIXME I dont think below implementation works for batch version
                # L2 loss
                # D_loss = -torch.mean(self.log(1 - d_fake_decision)) - torch.mean(self.log(d_real_decision))
                # D_loss.backward()

                d_optimizer.step()

            # train G
            for g_train_idx in range(g_steps):


    def backward(self):
        pass

    # def train(self, **kwargs):
    #     pass
    #
    # def test(self):
    #     pass

    # L2 loss instead of Binary cross entropy loss (this is optional for stable training)
    # FIXME: is L2 loss the same as MSELoss in torch loss module?
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
        epsilon = to_var(torch.rand(self.batch_size, 1))
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