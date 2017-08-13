
from __future__ import print_function
from __future__ import division

import sys
import os
import time
sys.path.append(os.path.abspath(__file__ + "/../../"))
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
from data_proc import *
# FIXME: had some problem importing util.py; importing successful but 
#        functions cannot be called (NameError: global name XXX is not defined)
#        fast solution: copied asMinutes and timeSince functions here
from util import *

import torch
from torch.autograd import Variable
from D_eval import *

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


######################################################################
# Training the Model
# context = input_variable
def train(discriminator, criterion, optimizer, batch_size, embeddings_size,
    embeddings_index, word2index, index2word, triplets,
    to_file, path_to_loss_f, path_to_sample_out_f, path_to_exp_out,
    n_iters=10, print_every=10, plot_every=100):

    begin_time = time.time()

    # open the files
    if to_file:
        loss_f = open(path_to_loss_f,'w+')
        sample_out_f = open(path_to_sample_out_f, 'w+')

    # plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    print()

    for iter in range(1, n_iters + 1):

        # prepare batch
        training_batch, seq_lens, fake_training_batch, fake_seq_lens = get_random_batch(triplets, batch_size, with_fake=True)
        # concat the context_ans batch with the question batch
        # each element in the training batch is context + question + answer
        training_batch, _, seq_lens = prepare_batch_var(training_batch, seq_lens, fake_training_batch, fake_seq_lens,
                                                        batch_size, word2index, embeddings_index, embeddings_size,
                                                        mode = ['word'], concat_opt='cqa', with_fake=True)

        train_input = Variable(training_batch[0].cuda()) if use_cuda else Variable(training_batch[0]) # embeddings vectors, size = [seq len x batch size x embedding dim]
        # the labels are the last element of training_batch; see prepare_batch_var in data_proc.py for detail
        train_label = Variable(torch.FloatTensor(training_batch[-1]).cuda()) if use_cuda else Variable(torch.FloatTensor(training_batch[-1]))

        optimizer.zero_grad()
        loss = 0
        outputs = discriminator.forward(train_input, train_label, seq_lens[0])
        loss += discriminator.backward(outputs, train_label, criterion, optimizer)

        print_loss_total += loss.data[0]
        plot_loss_total += loss.data[0]

        # log on console
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(begin_time, iter / float(n_iters)),
                                         iter, iter / n_iters * 100, print_loss_avg))
            evaluate(discriminator, triplets, word2index, embeddings_index, embeddings_size, eval_batch_size=100)
            print('-------------------------------')
            print('-------------------------------')
            print()

        # save error to file for plotting later
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            # plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            if to_file:
                loss_f.write(unicode(plot_loss_avg))
                loss_f.write(unicode('\n'))

    # showPlot(plot_losses)
    if to_file:
        loss_f.close()



