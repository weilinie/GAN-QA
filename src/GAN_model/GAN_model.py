import sys, os
sys.path.append(os.path.abspath(__file__ + "/../../") + '/G_c_a_sep')
sys.path.append(os.path.abspath(__file__ + "/../../") + '/D_baseline')
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
from data_proc import *
from G_c_a_sep import G
from D_model import *
from G_eval import *

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
##################################################################

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
                 use_attn, batch_size, pretrain=False, G_path=None
                 ):

        super(GAN_model, self).__init__()

        self.G = G(G_enc_input_size, G_enc_hidden_size, G_enc_n_layers, G_enc_num_directions, G_dec_input_size,
                   G_dec_hidden_size, G_output_size, G_dec_n_layers, G_dec_num_directions, batch_size)
        if pretrain:
            # load the G model from G_path
            self.G = torch.load(G_path)

        self.D = D(D_enc_input_size, D_enc_hidden_size, D_enc_n_layers, D_num_directions, D_mlp_hidden_size,
                   D_num_attn_weights, D_mlp_output_size, use_attn, batch_size)

    def train(self, triplets, n_iters, d_steps, d_optimizer, g_steps, g_optimizer, batch_size, max_len,
              criterion, word2index, index2word, embeddings_index, embeddings_size, print_every,
              to_file = False, loss_f = None, sample_out_f = None):
        # criterion is for both G and D

        # record start time for logging
        begin_time = time.time()

        for iter in range(1, n_iters + 1):

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

                train_input = Variable(cqa_batch[0].cuda()) if use_cuda else Variable(
                    cqa_batch[0])  # embeddings vectors, size = [seq len x batch size x embedding dim]

                d_real_decision = self.D.forward(train_input, cqa_lens[0])
                real_target = Variable(torch.FloatTensor([1]*batch_size)).cuda() if use_cuda else \
                    Variable(torch.FloatTensor([1]*batch_size))
                # d_real_error = criterion(d_real_decision, real_target)  # ones = true
                # d_real_error.backward()  # compute/store gradients, but don't change params

                #  1B: Train D on fake
                fake_cqa_batch, _, fake_cqa_lens = prepare_fake_batch_var(self.G, training_batch, max_len, batch_size,
                                                                          word2index, index2word, embeddings_index,
                                                                          embeddings_size, mode = ('word'))
                d_fake_data = Variable(fake_cqa_batch[0].cuda()) if use_cuda else Variable(fake_cqa_batch[0])
                d_fake_decision = self.D.forward(d_fake_data, fake_cqa_lens[0])
                fake_target = Variable(torch.FloatTensor([0]*batch_size)).cuda() if use_cuda else \
                    Variable(torch.FloatTensor([0]*batch_size))
                # d_fake_error = criterion(d_fake_decision, fake_target)  # zeros = fake
                # d_fake_error.backward()
                # d_optimizer.step()

                # accumulate loss
                # FIXME I dont think below implementation works for batch version
                # W_GAN loss
                d_error = torch.mean(d_fake_decision) - torch.mean(d_real_decision)
                d_error.backward()
                d_optimizer.step()

            # train G
            for g_train_idx in range(g_steps):
                self.G.zero_grad()

                # conditional data for generator
                training_batch, seq_lens = get_random_batch(triplets, batch_size)
                fake_cqa_batch, _, fake_cqa_lens = prepare_fake_batch_var(self.G, training_batch, max_len, batch_size,
                                                                          word2index, index2word, embeddings_index,
                                                                          embeddings_size, mode=('word'), detach=False)
                g_fake_data = Variable(fake_cqa_batch[0].cuda()) if use_cuda else Variable(fake_cqa_batch[0])
                dg_fake_decision = self.D.forward(g_fake_data, fake_cqa_lens[0])
                target = Variable(torch.FloatTensor([1]*batch_size).cuda()) if use_cuda else \
                    Variable(torch.FloatTensor([1]*batch_size))
                # g_error = criterion(dg_fake_decision, target)
                g_error = -torch.mean(dg_fake_decision)
                g_error.backward()
                g_optimizer.step()  # Only optimizes G's parameters

            # log error
            if iter % print_every == 0:
                if not to_file:
                    print('%s (%d %d%%)' % (timeSince(begin_time, iter / float(n_iters)), iter, iter / n_iters * 100))
                    # print("errors: D: real-%s/fake-%s G: %s " % ( d_real_error.data[0], d_fake_error.data[0], g_error.data[0]) )
                    print("errors: D: %s G: %s " % (d_error.data[0], g_error.data[0]))
                    print('---sample generated question---')
                    # sample a triple and print the generated question
                    evaluate(self.G, triplets, embeddings_index, embeddings_size, word2index, index2word, max_len)
                else:
                    print('%s (%d %d%%)' % (timeSince(begin_time, iter / float(n_iters)), iter, iter / n_iters * 100))
                    loss_f.write(unicode('%s (%d %d%%)\n' % (timeSince(begin_time, iter / float(n_iters)), iter, float(iter) / float(n_iters) * 100)))
                    # loss_f.write(unicode("errors: D: real-%s/fake-%s G: %s \n" % ( d_real_error.data[0], d_fake_error.data[0], g_error.data[0])))
                    loss_f.write(unicode("errors: D: %s G: %s " % (d_error.data[0], g_error.data[0])))
                    loss_f.write(unicode('\n'))
                    sample_out_f.write(unicode('%s (%d %d%%)\n' % (timeSince(begin_time, iter / float(n_iters)), iter, float(iter) / float(n_iters) * 100)))
                    evaluate(self.G, triplets, embeddings_index, embeddings_size, word2index, index2word, max_len,
                             to_file, sample_out_f)
                    sample_out_f.write(unicode('\n'))

    # def train(self, **kwargs):
    #     pass
    #
    # def test(self):
    #     pass

    # L2 loss instead of Binary cross entropy loss (this is optional for stable training)
    # FIXME: is L2 loss the same as MSELoss in torch loss module?
    # FIXME: these losses don't work with minibatch yet?
    def loss(self, D_real, D_fake, gen_params, disc_params, cond_real_data, cond_fake_data, mode, lr=None):
        mode = mode.lower()
        if mode == 'gan':
            G_loss = -torch.mean(self.log(D_fake))
            # FIXME G_loss.backward()
            D_loss = -torch.mean(self.log(1 - D_fake)) - torch.mean(self.log(D_real))
            # FIXME D_loss.backward()
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



# same context and answer as in the real batch, but generated question
def prepare_fake_batch_var(generator, batch, max_len, batch_size, word2index, index2word,
                           embeddings_index, embeddings_size, mode = ('word'), detach=True):

    batch_vars = []
    batch_var_orig = []

    cqa = []
    cqa_len = []
    labels = torch.LongTensor([0] * batch_size) # all fake labels, thus all 0's
    for b in range(batch_size):
        ca = batch[0][b] + batch[2][b]
        fake_q_sample = G_sampler(generator, ca, embeddings_index, embeddings_size, word2index, index2word, max_len, detach=detach)
        cqa.append(batch[0][b] + fake_q_sample + batch[2][b])
        cqa_len.append(len(batch[0][b] + fake_q_sample + batch[2][b]))

    batch = [cqa, batch[3], batch[4], labels]
    seq_lens = [cqa_len]

    # sort this batch_var in descending order according to the values of the lengths of the first element in batch
    num_batch = len(batch)
    all = batch + seq_lens
    all = sorted(zip(*all), key=lambda p: len(p[0]), reverse=True)
    all = zip(*all)
    batch = all[0:num_batch]
    seq_lens = all[num_batch:]
    batch_orig = batch

    for b in range(num_batch):

        batch_var = batch[b]

        # if element in batch is float, i.e. indices, then do nothing
        if isinstance(batch_var[0], int):
            batch_var = list(batch_var)
            pass
        else:
            # pad each context, question, answer to their respective max length
            if mode[b]  == 'index':
                batch_padded = [pad_sequence(s, max(seq_lens[b]), word2index, mode='index') for s in batch_var]
            else:
                batch_padded = [pad_sequence(s, max(seq_lens[b]), word2index) for s in batch_var]

            # init variable matrices
            if mode[b] == 'index':
                batch_var = torch.LongTensor(max(seq_lens[b]), batch_size) # long tensor for module loss criterion
            else:
                batch_var = torch.FloatTensor(max(seq_lens[b]), batch_size, embeddings_size)

            # FIXME: very stupid embedded for loop implementation
            for i in range(batch_size):
                for j in range(max(seq_lens[b])):
                    if mode[b] == 'index':
                        batch_var[j, i] = batch_padded[i][j]
                    else:
                        batch_var[j, i,] = embeddings_index[batch_padded[i][j]]

        batch_vars.append(batch_var)

    # the second output is for debugging purpose
    return batch_vars, batch_orig, seq_lens
