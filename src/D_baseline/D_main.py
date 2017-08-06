# -*- coding: utf-8 -*-

# NOTE: this is NOT tensorflow. This is PyTorch implementation, standalone of GAN.

"""
question answering model baseline

code adapted from <https://github.com/spro/practical-pytorch>`_

made use of seq2seq learning <http://arxiv.org/abs/1409.3215>
and attention mechanism <https://arxiv.org/abs/1409.0473>

input: a paragraph (aka context), and an answer, both represented by a sequence of tokens
output: a question, represented by a sequence of tokens

"""

#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# requirements
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
from __future__ import unicode_literals, print_function, division
import time
from io import open
import unicodedata
import string
import re
import random

import spacy
from spacy.en import English
spacynlp = English()

import torch

import nltk
import json
import numpy as np
import os

use_cuda = torch.cuda.is_available()

#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# evaluation script
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#


######### set paths
# default values for the dataset and the path to the project/dataset
dataset = 'squad'
f_name = 'train-v1.1.json'
path_to_dataset = '/home/jack/Documents/QA_QG/data/'
path_to_data = path_to_dataset + dataset + '/' + f_name
GLOVE_DIR = path_to_dataset + 'glove.6B/'
# path for experiment outputs
exp_name = 'QG_seq2seq_baseline'
path_to_exp_out = '/home/jack/Documents/QA_QG/exp_results/' + exp_name
loss_f = 'loss.txt'
sample_out_f = 'sample_outputs.txt'
path_to_loss_f = path_to_exp_out + '/' + loss_f
path_to_sample_out_f = path_to_exp_out + '/' + sample_out_f

######### first load the pretrained word embeddings
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    coefs = torch.from_numpy(coefs)
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# get dimension from a random sample in the dict
embeddings_size = random.sample( embeddings_index.items(), 1 )[0][1].size(-1)
SOS_token = -torch.ones(embeddings_size) # start of sentence token, all zerons
EOS_token = torch.ones(embeddings_size) # end of sentence token, all ones
UNK_token = torch.ones(embeddings_size) + torch.ones(embeddings_size) # these choices are pretty random
# add special tokens to the embeddings
embeddings_index['SOS'] = SOS_token
embeddings_index['EOS'] = EOS_token
embeddings_index['UNK'] = UNK_token


######### read corpus
triplets = readSQuAD(path_to_data)

######### corpus preprocessing
# TODO: need some work here: deal with inprecise tokenizer, 
# words that do not appear in embeddings, etc

## find all unique tokens in the data (should be a subset of the number of embeddings)
data_tokens = []
for triple in triplets:
    c = [token.string.strip() for token in spacynlp.tokenizer(triple[0])]
    q = [token.string.strip() for token in spacynlp.tokenizer(triple[1])]
    a = [token.string.strip() for token in spacynlp.tokenizer(triple[2])]
    data_tokens += c + q + a
data_tokens = list(set(data_tokens)) # find unique
data_tokens = ['SOS', 'EOS', 'UNK'] + data_tokens
print(data_tokens[0:20])
# experimental usage only
data_tokens = data_tokens[0:10000]

num_tokens = len(data_tokens)
# generate some index
# token_indices = random.sample(range(0, len(data_tokens)), 20)
# # debugging purpose
# token_subset = [data_tokens[i] for i in token_indices]
# print('original tokens: ' + str(token_subset))
# # extra preprocessing step to replace all tokens in data_tokens 
# # that does not appear in embeddings_index to 'UNK'
# # OOV_indices = [i for i, e in enumerate(data_tokens) if e not in set(embeddings_index.keys())] # indices of out of vocabulary words in data_tokens
# for i in OOV_indices:
#     data_tokens[i] = 'UNK'
# # debugging: randomly sample 20 tokens from data_tokens. shouldn't be all UNK
# token_subset = [data_tokens[i] for i in token_indices]
# print('modified tokens: ' + str(token_subset))

# build word2index dictionary and index2word dictionary
word2index = {}
index2word = {}
for i in range(0, len(data_tokens)):
    index2word[i] = data_tokens[i]
    word2index[data_tokens[i]] = i

print('reading and preprocessing data complete.')
print('found %s unique tokens in corpus.' % len(data_tokens))
if use_cuda:
    print('GPU ready.')
print('')
print('start training...')
print('')


######### set up model
hidden_size1 = 64
hidden_size2 = 64
# context encoder
encoder1 = EncoderRNN(embeddings_size, hidden_size1)
# answer encoder
encoder2 = EncoderRNN(embeddings_size, hidden_size2)
# decoder
attn_decoder1 = AttnDecoderRNN(embeddings_size, hidden_size1, num_tokens, 
                                1, dropout_p=0.1)

if use_cuda:
    t1 = time.time()
    encoder1 = encoder1.cuda()
    t2 = time.time()
    print('time load encoder 1: ' + str(t2 - t1))
    encoder2 = encoder2.cuda()
    t3 = time.time()
    print('time load encoder 2: ' + str(t3 - t2))
    attn_decoder1 = attn_decoder1.cuda()
    t4 = time.time()
    print('time load decoder: ' + str(t4 - t3))


######### start training
trainIters(encoder1, encoder2, attn_decoder1, embeddings_index, word2index, data_tokens,
            path_to_loss_f, path_to_sample_out_f, path_to_exp_out,
            75000, print_every=1)

# save the final model
torch.save(encoder1, path_to_exp_out+'/encoder1.pth')
torch.save(encoder2, path_to_exp_out+'/encoder2.pth')
torch.save(decoder, path_to_exp_out+'/decoder.pth')

