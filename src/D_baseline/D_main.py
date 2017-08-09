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
from __future__ import print_function
from __future__ import division

import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
# print(os.path.abspath(__file__ + '/../../../../')+'/util')
sys.path.append(os.path.abspath(__file__ + "/../../") + '/G_baseline_batch')
from data_proc import *

# from ..util.data_proc import *
from model_zoo import *
from G_train_batch import *
# from G_eval_batch import *
import numpy as np

use_cuda = torch.cuda.is_available()


######### set paths
# TODO: to run properly, change the following paths and filenames
# default values for the dataset and the path to the project/dataset
dataset = 'squad'
f_name = 'dev-v1.1.json'
path_to_dataset = os.path.abspath(__file__ + '/../../../../') + '/data/'
path_to_data = path_to_dataset + dataset + '/' + f_name
GLOVE_DIR = path_to_dataset + 'glove.6B/'
# path for experiment outputs
# exp_name = 'QG_seq2seq_baseline'
path_to_exp_out = os.path.abspath(__file__ + '/../../../../') + '/exp_results_temp/'
loss_f = 'loss_temp.txt'
sample_out_f = 'sample_outputs_temp.txt'
path_to_loss_f = path_to_exp_out + '/' + loss_f
path_to_sample_out_f = path_to_exp_out + '/' + sample_out_f


######### first load the pretrained word embeddings
path_to_glove = os.path.join(GLOVE_DIR, 'glove.6B.50d.txt')
embeddings_index, embeddings_size = readGlove(path_to_glove)


######### read corpus
raw_triplets = read_raw_squad(path_to_data)
triplets = tokenize_squad(raw_triplets, embeddings_index)

# find max length of context, question, answer, respectively
max_len_c, max_len_q, max_len_a = max_length(triplets)

######### corpus preprocessing
# words that do not appear in embeddings, etc

## find all unique tokens in the data (should be a subset of the number of embeddings)
effective_tokens, effective_num_tokens = count_effective_num_tokens(triplets, embeddings_index)
print('effective number of tokens: ' + str(effective_num_tokens))
print('expected initial loss: ' + str(-np.log(1/float(effective_num_tokens))) + '\n')
# build word2index dictionary and index2word dictionary
word2index, index2word = generate_look_up_table(effective_tokens, effective_num_tokens)


######### set up model
enc_hidden_size = 64
mlp_hidden_size = 64
output_size = 1
num_attn_weights = 1000
batch_size = 40
# context encoder
encoder = EncoderRNN(embeddings_size, enc_hidden_size, batch_size)
# decoder
mlp = MLP(mlp_hidden_size, output_size, encoder, num_attn_weights)

if use_cuda:
    t1 = time.time()
    encoder = encoder.cuda()
    t2 = time.time()
    print('time load encoder 1: ' + str(t2 - t1))
    mlp = mlp.cuda()
    t3 = time.time()
    print('time load decoder: ' + str(t3 - t2))


######### start training
trainIters(encoder, mlp, batch_size, embeddings_size,
           embeddings_index, word2index, index2word, triplets,
           path_to_loss_f, path_to_sample_out_f, path_to_exp_out,
           n_iters = 5, print_every=1, plot_every=1, learning_rate=0.001)

# save the final model
torch.save(encoder1, path_to_exp_out+'/encoder1.pth')
torch.save(encoder2, path_to_exp_out+'/encoder2.pth')
torch.save(decoder, path_to_exp_out+'/decoder.pth')





