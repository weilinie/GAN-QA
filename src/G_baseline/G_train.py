
from __future__ import print_function
from __future__ import division

import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../"))
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
from data_proc import *
# FIXME: had some problem importing util.py; importing successful but 
#        functions cannot be called (NameError: global name XXX is not defined)
#        fast solution: copied asMinutes and timeSince functions herefrom util import *
from G_eval import *

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import time

use_cuda = torch.cuda.is_available()


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



def trainIters(generator, optimizer, batch_size, embeddings_size,
    embeddings_index, word2index, index2word, max_length, triplets, teacher_forcing_ratio,
    to_file, loss_f, sample_out_f, path_to_exp_out,
    n_iters=5, print_every=10, plot_every=100):

    begin_time = time.time()

    # plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    print()

    for iter in range(1, n_iters + 1):

        # prepare batch
        training_batch, seq_lens = get_random_batch(triplets, batch_size)
        training_batch, _, seq_lens = prepare_batch_var(training_batch, seq_lens, batch_size, word2index, embeddings_index, embeddings_size, use_cuda=1, mode=['word', 'index'], concat_opt='ca')
        inputs_ca = Variable(training_batch[0].cuda()) if use_cuda else Variable(training_batch[0]) # embeddings vectors, size = [seq len x batch size x embedding dim]
        inputs_q = Variable(training_batch[1].cuda()) if use_cuda else Variable(training_batch[1]) # represented as indices, size = [seq len x batch size]

        max_c_a_len = max(seq_lens[0])  # max seq length of context + ans combined
        max_q_len = max(seq_lens[1])  # max seq length of question

        optimizer.zero_grad()
        loss = 0
        all_decoder_outputs = generator.forward(inputs_ca, inputs_q, seq_lens[0], batch_size, max_q_len,
                                                embeddings_index, embeddings_size, word2index, index2word,
                                                teacher_forcing_ratio)
        loss += generator.backward(all_decoder_outputs, inputs_q, seq_lens[1], optimizer)

        print_loss_total += loss.data[0]
        plot_loss_total += loss.data[0]

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(begin_time, iter / float(n_iters)),
                                         iter, iter / n_iters * 100, print_loss_avg))
            print('---sample generated question---')
            # sample a triple and print the generated question
            evaluate(generator, triplets, embeddings_index, embeddings_size, word2index, index2word, max_length)
            print('-------------------------------')
            print('-------------------------------')
            print()

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            # plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            if to_file:
                loss_f.write(unicode('%s (%d %d%%)\n' % (timeSince(begin_time, iter / float(n_iters)), iter, float(iter) / float(n_iters) * 100)))
                loss_f.write(unicode(plot_loss_avg))
                loss_f.write(unicode('\n'))
                sample_out_f.write(unicode('%s (%d %d%%)\n' % (timeSince(begin_time, iter / float(n_iters)), iter, float(iter) / float(n_iters) * 100)))
                evaluate(generator, triplets, embeddings_index, embeddings_size, word2index, index2word, max_length, to_file, sample_out_f)
                sample_out_f.write(unicode('\n'))

                    

    # showPlot(plot_losses)
    if to_file:
        loss_f.close()



