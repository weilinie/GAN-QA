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

from ..util.data_proc import *
from G_eval_batch import *

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
def train(context_ans_batch_var, question_batch_var, batch_size,
          embeddings_index, word2index, index2word, teacher_forcing_ratio,
          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # get max lengths of (context + answer) and question
    max_c_a_len = context_ans_batch_var.size(0) # max seq length of context + ans combined
    max_q_len = question_batch_var.size(0) # max seq length of question

    # get lengths of each (context + answer) and question in each batch
    # (by simply read one of the inputs to this function)
    # TODO figure out if the below two lines are needed
    # c_a_lens = seq_lens[0]
    # q_lens = seq_lens[1]

    loss = 0

    # context encoding
    # output size: (seq_len, batch, hidden_size)
    # hidden size: (num_layers, batch, hidden_size)
    # the collection of all hidden states per batch is of size (seq_len, batch, hidden_size * num_directions)
    encoder_hiddens, encoder_hidden = encoder(context_ans_batch_var, None)

    # decoder
    # prepare decoder inputs as word embeddings in a batch
    # decoder_input size: (1, batch size, embedding size); first dim is 1 because only one time step;
    # nee to have a 3D tensor for input to nn.GRU module
    decoder_input = Variable( embeddings_index['SOS'].repeat(batch_size, 1).unsqueeze(0) )

    # use teacher forcing to step through each token in the decoder sequence
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(max_q_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, encoder_hiddens, embeddings_index)

            # accumulate loss
            loss += criterion(decoder_output[0], Variable(question_batch_var[di]))

            # change next time step input to current target output, in embedding format
            for b in range(batch_size):
                decoder_input[1, b] = Variable(embeddings_index[index2word[question_batch_var[di, b]]].cuda) if use_cuda else\
                                      Variable(embeddings_index[index2word[question_batch_var[di, b]]])# Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_q_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, encoder_hiddens, embeddings_index)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(embeddings_index[index2word[di]].cuda()) if use_cuda else \
                            Variable(embeddings_index[index2word[di]])

            # accumulate loss
            loss += criterion(decoder_output[0], Variable(question_batch_var[di]))
            if ni == word2index['EOS']:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / ( float(max_q_len) * float(batch_size) )



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

def trainIters(encoder, decoder, batch_size, embeddings_size,
    embeddings_index, word2index, index2word, max_length, triplets, teacher_forcing_ratio,
    path_to_loss_f, path_to_sample_out_f, path_to_exp_out,
    n_iters, print_every=10, plot_every=100, learning_rate=0.01):

    begin_time = time.time()

    # open the files
    loss_f = open(path_to_loss_f,'w+') 
    sample_out_f = open(path_to_sample_out_f, 'w+')

    # plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    print()

    for iter in range(1, n_iters + 1):

        # prepare batch
        training_batch, seq_lens = get_random_batch(triplets, batch_size, word2index)
        training_batch = prepare_batch_var(training_batch, seq_lens, batch_size, embeddings_index, embeddings_size)
        context_ans_batch_var = training_batch[0] # embeddings vectors, size = [seq len x batch size x embedding dim]
        question_batch_var = training_batch[1] # represented as indices, size = [seq len x batch size]

        # prepare encoder input
        context_ans_batch_var = nn.utils.rnn.pack_padded_sequence(context_ans_batch_var, seq_lens[0])
        
        start = time.time()
        loss = train(context_ans_batch_var, question_batch_var, seq_lens, batch_size,
                     embeddings_index, word2index, index2word, teacher_forcing_ratio,
                     encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
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
            evaluateRandomly(encoder1, encoder2, decoder, triplets, embeddings_index, word2index, index2word, max_length, n=1)
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



