#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# training and evaluation
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import time

# from ..util.data_proc import *

import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
# print(os.path.abspath(__file__ + '/../../../../')+'/util')
sys.path.append(os.path.abspath(__file__ + "/../../") + '/G_baseline_batch')
from data_proc import *
from util import *

from D_baseline_model import *
from D_eval import *

use_cuda = torch.cuda.is_available()

######################################################################
# Training the Model
# ------------------
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability

# context = input_variable
def train(train_batch, batch_size, seq_lens, true_labels,
          embeddings_index, embeddings_size, word2index, index2word,
          encoder, mlp, encoder_optimizer, mlp_optimizer, criterion):

    encoder_optimizer.zero_grad()
    mlp_optimizer.zero_grad()

    # get max lengths of (context + answer) and question
    #max_c_a_len = max(seq_lens[0]) # max seq length of context + ans combined
    #max_q_len = max(seq_lens[1]) # max seq length of question

    loss = 0

    # context encoding
    # output size: (seq_len, batch, hidden_size)
    # hidden size: (num_layers, batch, hidden_size)
    # the collection of all hidden states per batch is of size (seq_len, batch, hidden_size * num_directions)
    # print(train_batch.size())
    # print(len(seq_lens))
    # print(type(seq_lens))
    # print(seq_lens)
    encoder_hiddens, encoder_hidden = encoder(train_batch, seq_lens, None)

    # outputs = F.sigmoid(mlp(encoder_hiddens))
    outputs = mlp(encoder_hiddens)

    loss += criterion(outputs, true_labels)

    loss.backward()
    encoder_optimizer.step()
    mlp_optimizer.step()

    # return loss
    # FIXME: figure out if loss need to be divided by batch_size
    return loss.data[0] # did not divide by batch size for now



######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

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
        # c_a_q = list(training_batch[0])
        # concat the context_ans batch with the question batch
        # each element in the training batch is context + question + answer
        training_batch, _, seq_lens = prepare_batch_var(training_batch, seq_lens, fake_training_batch, fake_seq_lens,
                                                        batch_size, word2index, embeddings_index, embeddings_size,
                                                        mode = ['word', 'index'], concat_opt='cqa', with_fake=True)

        # prints for debug use
        # print(training_batch[0].size())
        # print(type(seq_lens[0]))
        # print(seq_lens[0])

        train_input = Variable(training_batch[0].cuda()) if use_cuda else Variable(training_batch[0]) # embeddings vectors, size = [seq len x batch size x embedding dim]
        train_label = Variable(torch.FloatTensor(training_batch[-1]).cuda()) if use_cuda else Variable(torch.FloatTensor(training_batch[-1]))

        start = time.time()

        loss = train(train_input, batch_size, seq_lens[0], train_label,
                     embeddings_index, embeddings_size, word2index, index2word,
                     encoder, mlp, encoder_optimizer, mlp_optimizer, criterion)
        # print('loss at iteration ' + str(iter) + ' is: ' + str(loss))

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
            evaluate(encoder, mlp, triplets, word2index, embeddings_index, embeddings_size, eval_batch_size=10)
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



