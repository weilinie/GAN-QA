import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../"))
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
from data_proc import *
from util import *

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import time

use_cuda = torch.cuda.is_available()


# max_length constrains the maximum length of the generated question
def evaluate(generator, triplets, embeddings_index, embeddings_size, word2index, index2word, max_length):

    # prepare test input
    batch_size = 1
    training, seq_lens = get_random_batch(triplets, batch_size)
    context_words = training[0]
    answer_words = training[2]
    question_words = training[1]
    fake_batch = None
    fake_seq_lens = None
    training, _, seq_lens = prepare_batch_var(training, seq_lens, fake_batch, fake_seq_lens, batch_size, word2index, embeddings_index, embeddings_size, mode=['word', 'index'], concat_opt='ca')
    inputs_ca = Variable(training[0].cuda()) if use_cuda else Variable(training[0]) # embeddings vectors, size = [seq len x batch size x embedding dim]
    inputs_q = Variable(training[1].cuda()) if use_cuda else Variable(training[1]) # represented as indices, size = [seq len x batch size]

    all_decoder_outputs = generator.forward(inputs_ca, inputs_q, seq_lens[0], batch_size, max_length,
                                            embeddings_index, embeddings_size, word2index, index2word,
                                            teacher_forcing_ratio=0)

    decoded_words = []
    # get the word token and add to the list of words
    for di in range(max_length):
        # top value and index of every batch
        topv, topi = all_decoder_outputs[di].data.topk(1)
        ni = topi[0][0]
        if (ni == word2index['EOS']) or (ni == word2index['PAD']):
            decoded_words.append('EOS')
            # decoder_attentions[di] = decoder_attention[0].data
            break
        else:
            decoded_words.append(index2word[ni])

    # # context (paragraph + answer) encoding
    # encoder_hiddens, encoder_hidden = encoder(context_ans_var, seq_lens[0], None)
    #
    # # prepare decoder input
    # decoder_input = Variable(embeddings_index['SOS'].repeat(batch_size, 1).unsqueeze(0))
    # if use_cuda:
    #     decoder_input = decoder_input.cuda()
    #
    # # decoder generate words
    # decoded_words = []
    # decoder_attentions = torch.zeros(max_length, encoder_hiddens.size()[0])
    # for di in range(max_length):
    #     decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, encoder_hiddens, embeddings_index)
    #
    #     # top value and index of every batch
    #     # size of both topv, topi = (batch size, 1)
    #     topv, topi = decoder_output.data.topk(1)
    #     ni = topi[0][0]
    #
    #     # get the word token and add to the list of words
    #     if (ni == word2index['EOS']) or (ni == word2index['PAD']):
    #         decoded_words.append('EOS')
    #         decoder_attentions[di] = decoder_attention[0].data
    #         break
    #     else:
    #         decoded_words.append(index2word[ni])
    #
    #     # prepare decoder next time step input
    #     # get the output word for every batch
    #     decoder_input = Variable(torch.FloatTensor(1, batch_size, embeddings_size).cuda()) if use_cuda else \
    #         Variable(torch.FloatTensor(1, batch_size, embeddings_size))
    #     decoder_input[0, 0] = embeddings_index[index2word[topi[0][0]]].cuda() if use_cuda else \
    #                           embeddings_index[index2word[topi[0][0]]]

    # print results
    print('context              > ' + ' '.join(context_words[0]))
    print('answer               > ' + ' '.join(answer_words[0]))
    print('question             > ' + ' '.join(question_words[0]))
    true_q = []
    for i in range(seq_lens[1][0]):
        true_q.append(index2word[inputs_q[i][0].data[0]])
    print('question with padding> ' + ' '.join(true_q))
    print('generated question   > ' + ' '.join(decoded_words))

    # return decoded_words, decoder_attentions

