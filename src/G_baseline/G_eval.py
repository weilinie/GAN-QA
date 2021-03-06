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
def evaluate(generator, triplets, embeddings_index, embeddings_size, word2index, index2word, max_length,
             to_file = False, sample_out_f = None):

    # prepare test input
    batch_size = 1
    training, seq_lens = get_random_batch(triplets, batch_size)
    context_words = training[0]
    answer_words = training[2]
    question_words = training[1]
    training, _, seq_lens = prepare_batch_var(training, seq_lens, batch_size, word2index, embeddings_index, embeddings_size, mode=['word', 'index'], concat_opt='ca')
    inputs_ca = Variable(training[0].cuda()) if use_cuda else Variable(training[0]) # embeddings vectors, size = [seq len x batch size x embedding dim]
    # inputs_q = Variable(training[1].cuda()) if use_cuda else Variable(training[1]) # represented as indices, size = [seq len x batch size]
    inputs_q = None

    all_decoder_outputs = generator.forward(inputs_ca, inputs_q, seq_lens[0], batch_size, max_length,
                                            embeddings_index, embeddings_size, word2index, index2word,
                                            teacher_forcing_ratio=0)

    decoded_sentences = []
    decoded_words = []
    for b in range(batch_size):
        # get the word token and add to the list of words
        for di in range(max_length):
            # top value and index of every batch
            topv, topi = all_decoder_outputs[di,b].data.topk(1)
            ni = topi[0]
            if (ni == word2index['EOS']) or (ni == word2index['PAD']):
                decoded_words.append('EOS')
                # decoder_attentions[di] = decoder_attention[0].data
                break
            else:
                decoded_words.append(index2word[ni])
        decoded_sentences.append(decoded_words)

    # print results
    if not to_file:
        print('context              > ' + ' '.join(context_words[0]).encode('utf-8').strip())
        print('answer               > ' + ' '.join(answer_words[0]).encode('utf-8').strip())
        print('question             > ' + ' '.join(question_words[0]).encode('utf-8').strip())
        # true_q = []
        # for i in range(seq_lens[1][0]):
        #     true_q.append(index2word[inputs_q[i][0].data[0]])
        # print('question with padding> ' + ' '.join(true_q))
        print('generated question   > ' + ' '.join(decoded_words))
    else:
        sample_out_f.write(unicode('context              > ' + ' '.join(context_words[0]) + '\n'))
        sample_out_f.write(unicode('answer               > ' + ' '.join(answer_words[0]) + '\n'))
        sample_out_f.write(unicode('question             > ' + ' '.join(question_words[0]) + '\n'))
        sample_out_f.write(unicode('generated question   > ' + ' '.join(decoded_words) + '\n'))

    # TODO: uncomment the following return if you want to record the decoder outputs in file
    #       (note: need to modify this function call in G_train.py)
    # return decoded_sentences


def G_sampler(generator, ca, embeddings_index, embeddings_size, word2index, index2word, max_length):
# NOTE currently only generate one question at a time. multiple questions not yet supported

    var = torch.FloatTensor(len(ca), embeddings_size)
    for j in range(len(ca)):
        var[j] = embeddings_index[ca[j]]
    var = var.unsqueeze(1)
    if use_cuda:
        var = Variable(var.cuda())
    else:
        var = Variable(var)

    decoder_output = generator.forward(var, None, [len(ca)], 1, max_length,
                                       embeddings_index, embeddings_size, word2index, index2word,
                                       teacher_forcing_ratio=0).detach()
    decoder_output = decoder_output.squeeze(1)

    decoded_words = []
    for di in range(max_length):
        # top value and index of every batch
        topv, topi = decoder_output[di].data.topk(1)
        ni = topi[0]
        if (ni == word2index['EOS']) or (ni == word2index['PAD']):
            decoded_words.append('EOS')
            # decoder_attentions[di] = decoder_attention[0].data
            break
        else:
            decoded_words.append(index2word[ni])

    return decoded_words

