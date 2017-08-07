import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import time

import sys
sys.path.append('/home/jack/Documents/QA_QG/GAN-QA/src/util')
sys.path.append('/home/jack/Documents/QA_QG/GAN-QA/src/G_baseline_batch')
from data_proc import *
from util import *

# from ..util.data_proc import *

use_cuda = torch.cuda.is_available()


######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#
# max_length constrains the maximum length of the generated question
def evaluate(encoder, decoder, triplets, embeddings_index, embeddings_size, word2index, index2word, max_length):

    # prepare test input
    batch_size = 1
    training, seq_lens = get_random_batch(triplets, batch_size, word2index)
    context_words = training[0]
    training = prepare_batch_var(training, seq_lens, batch_size, embeddings_index, embeddings_size)
    context_ans_var = training[0]  # embeddings vectors, size = [seq len x batch size x embedding dim]
    question_var = training[1]  # represented as indices, size = [seq len x batch size]

    # context (paragraph + answer) encoding
    encoder_hiddens, encoder_hidden = encoder(context_ans_var, seq_lens[0], None)

    # prepare decoder input
    decoder_input = Variable(embeddings_index['SOS'].repeat(batch_size, 1).unsqueeze(0))
    if use_cuda:
        decoder_input = decoder_input.cuda()

    # decoder generate words
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, encoder_hiddens.size()[0])
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, encoder_hiddens, embeddings_index)
        # print('decoder attention dimension: ' + str(decoder_attention.size()))

        # top value and index of every batch
        # size of both topv, topi = (batch size, 1)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        # get the word token and add to the list of words
        if (ni == word2index['EOS']) or (ni == word2index['PAD']):
            decoded_words.append('EOS')
            decoder_attentions[di] = decoder_attention[0].data
            break
        else:
            decoded_words.append(index2word[ni])

        # prepare decoder next time step input
        # get the output word for every batch
        decoder_input = Variable(torch.FloatTensor(1, batch_size, embeddings_size).cuda()) if use_cuda else \
            Variable(torch.FloatTensor(1, batch_size, embeddings_size))
        decoder_input[0, 0] = embeddings_index[index2word[topi[0][0]]].cuda() if use_cuda else \
                              embeddings_index[index2word[topi[0][0]]]

    # print results
    print('context and question > ' + ' '.join(context_words[0]))
    true_q = []
    for i in range(seq_lens[1][0]):
        true_q.append(index2word[question_var[i][0]])
    print('question             > ' + ' '.join(true_q))
    print('generated question   > ' + ' '.join(decoded_words))

    return decoded_words, decoder_attentions

