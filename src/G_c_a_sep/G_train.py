
import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../"))
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
# from util import timeSince, asMinutes, plotLoss
from data_proc import *
# FIXME: had some problem importing util.py; importing successful but 
#        functions cannot be called (NameError: global name XXX is not defined)
#        fast solution: copied asMinutes and timeSince functions herefrom util import *
from G_eval import *

import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


########################################################################################################################
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math

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
# show loss function
def plotLoss(loss_f, plot_every, save_path=None, from_file=True, f_name='loss.png', title='training loss'):
    if from_file:
        loss_vec = []
        with open(loss_f) as f:
            content = f.readlines()
            content = [x.strip() for x in content] # list of every line, each a string
            for line in content:
                try:
                    loss_vec.append(float(line))
                except ValueError:
                    pass
    else:
        loss_vec = loss_f
    # plot
    plt.figure()
    plt.title(title)
    plt.xlabel('training iterations')
    plt.ylabel('loss')
    plt.grid()
    plt.plot([x*plot_every for x in range(1, len(loss_vec)+1)], loss_vec)
    if save_path == None:
        plt.savefig(f_name)
    else:
        plt.savefig(save_path + '/' + f_name)
########################################################################################################################


def trainIters(generator, optimizer, batch_size, embeddings_size,
    embeddings_index, word2index, index2word, max_length, triplets, teacher_forcing_ratio,
    to_file, loss_f, sample_out_f, path_to_exp_out,
    n_iters=1, print_every=1, plot_every=1, checkpoint_every=1):

    begin_time = time.time()

    # plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    plot_loss_avgs = []

    print()

    for iter in range(1, n_iters + 1):

        # prepare batch
        training_batch, seq_lens = get_random_batch(triplets, batch_size)
        training_batch, _, seq_lens = prepare_batch_var(
            training_batch, seq_lens, batch_size, word2index, embeddings_index, embeddings_size)

        # print(type(training_batch))
        # print(type(training_batch[0]))

        # prepare inputs (load to cuda)
        inputs = []
        for var in training_batch:
            if not isinstance(var, list):
                inputs.append(Variable(var.cuda())) if use_cuda else inputs.append(Variable(var))
            # NOTE not currently appending start and end index to inputs because model does not use them.
            # NOTE if want to apend, make sure these are changed from list to LongTensor
            # else:
            #     inputs.append(Variable(var))

        max_c_a_len = max(seq_lens[0])  # max seq length of context + ans combined
        max_q_len = max(seq_lens[1])  # max seq length of question

        optimizer.zero_grad()
        loss = 0
        all_decoder_outputs = generator.forward(inputs, seq_lens, batch_size, max_q_len,
                                                embeddings_index, embeddings_size, word2index, index2word,
                                                teacher_forcing_ratio)
        loss += generator.backward(all_decoder_outputs, inputs[1], seq_lens[1], optimizer)

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
            if to_file:
                sample_out_f.write(unicode('%s (%d %d%%)\n' % (timeSince(begin_time, iter / float(n_iters)), iter, float(iter) / float(n_iters) * 100)))
                evaluate(generator, triplets, embeddings_index, embeddings_size, word2index, index2word, max_length, to_file, sample_out_f)
                sample_out_f.write(unicode('\n'))
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_loss_avgs.append(plot_loss_avg)
            # plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            if to_file:
                loss_f.write(unicode('%s (%d %d%%)\n' % (timeSince(begin_time, iter / float(n_iters)), iter, float(iter) / float(n_iters) * 100)))
                loss_f.write(unicode(plot_loss_avg))
                loss_f.write(unicode('\n'))
        if to_file and ((iter % checkpoint_every == 0) or (iter == n_iters)):
            checkpoint_fname = 'checkpoint_iter_' + str(iter) + '.pth.tar'
            state = {
                        'iteration': iter + 1,
                        'g_state_dict': generator.state_dict(),
                        'g_optimizer' : optimizer.state_dict(),
                    }
            torch.save(state, path_to_exp_out+'/'+checkpoint_fname)
            plotLoss(plot_loss_avgs, plot_every, save_path=path_to_exp_out, f_name='d_loss_itr_'+str(iter)+'.png',
                title='training loss', from_file=False)

    # showPlot(plot_losses)
    if to_file:
        loss_f.close()



