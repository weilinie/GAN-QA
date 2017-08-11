
from __future__ import print_function
from __future__ import division

import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../")
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
from data_proc import *
from util import *

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import time

from D_baseline import *
from D_eval import *

use_cuda = torch.cuda.is_available()

######################################################################
# Training the Model
# context = input_variable
def train(train_batch, batch_size, seq_lens, true_labels,
          embeddings_index, embeddings_size, word2index, index2word,
          encoder, mlp, encoder_optimizer, mlp_optimizer, criterion):

    encoder_optimizer.zero_grad()
    mlp_optimizer.zero_grad()

    loss = 0

    # context encoding
    # output size: (seq_len, batch, hidden_size)
    # hidden size: (num_layers, batch, hidden_size)
    # the collection of all hidden states per batch is of size (seq_len, batch, hidden_size * num_directions)

    #TODAY: combine encoder and mlp into a single class "D" and replace other files thereafter

    encoder_hiddens, encoder_hidden = encoder(train_batch, seq_lens, None)

    outputs = mlp(encoder_hiddens)

    loss += criterion(outputs, true_labels)

    loss.backward()
    encoder_optimizer.step()
    mlp_optimizer.step()

    # return loss
    # FIXME: figure out if loss need to be divided by batch_size
    return loss.data[0] # did not divide by batch size for now



######################################################################
# training helper function
# this function set the optimizer, loss criterion, and load minibatch

def trainIters(encoder, mlp, batch_size, embeddings_size,
    embeddings_index, word2index, index2word, triplets,
    path_to_loss_f, path_to_sample_out_f, path_to_exp_out,
    n_iters=10, print_every=10, plot_every=100, learning_rate=0.01):

    begin_time = time.time()

    # open the files
    loss_f = open(path_to_loss_f,'w+')
    sample_out_f = open(path_to_sample_out_f, 'w+')

    # plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    mlp_optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)

    #criterion = nn.BCEWithLogitsLoss() # binary loss
    criterion = nn.BCELoss() 

    print()

    for iter in range(1, n_iters + 1):

        # prepare batch
        # do not need the answer location for now (the second output from get_random_batch)
        training_batch, seq_lens, fake_training_batch, fake_seq_lens = get_random_batch(triplets, batch_size, with_fake=True)
        # concat the context_ans batch with the question batch
        # each element in the training batch is context + question + answer
        training_batch, _, seq_lens = prepare_batch_var(training_batch, seq_lens, fake_training_batch, fake_seq_lens,
                                                        batch_size, word2index, embeddings_index, embeddings_size,
                                                        mode = ['word', 'index'], concat_opt='cqa', with_fake=True)

        train_input = Variable(training_batch[0].cuda()) if use_cuda else Variable(training_batch[0]) # embeddings vectors, size = [seq len x batch size x embedding dim]
        train_label = Variable(torch.FloatTensor(training_batch[-1]).cuda()) if use_cuda else Variable(torch.FloatTensor(training_batch[-1]))

        start = time.time()
        loss = train(train_input, batch_size, seq_lens[0], train_label,
                     embeddings_index, embeddings_size, word2index, index2word,
                     encoder, mlp, encoder_optimizer, mlp_optimizer, criterion)
        end = time.time()

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(begin_time, iter / float(n_iters)),
                                         iter, iter / n_iters * 100, print_loss_avg))
            print('time for one training iteration: ' + str(end - start))
            print('---sample generated question---')
            # sample a triple and print the generated question
            evaluate(encoder, mlp, triplets, word2index, embeddings_index, embeddings_size, eval_batch_size=100)
            print('-------------------------------')
            print('-------------------------------')
            print()

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            # plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            loss_f.write(unicode(plot_loss_avg))
            loss_f.write(unicode('\n'))

    # showPlot(plot_losses)
    loss_f.close()



