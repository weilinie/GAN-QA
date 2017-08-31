from __future__ import print_function
from __future__ import division

import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../"))
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')

from G_train import *
from G_c_a_sep import *
import numpy as np

global use_cuda
use_cuda = torch.cuda.is_available()
teacher_forcing_ratio = 0.75 # default in original code is 0.5


######### set paths
# TODO: to run properly, change the following paths and filenames
# default values for the dataset and the path to the project/dataset
dataset = 'squad'
f_name = 'dev-v1.1.json'
path_to_dataset = '/home/jack/Documents/QA_QG/data/'
path_to_data = path_to_dataset + dataset + '/' + f_name
GLOVE_DIR = path_to_dataset + 'glove.6B/'


######### first load the pretrained word embeddings
path_to_glove = os.path.join(GLOVE_DIR, 'glove.6B.50d.txt')
embeddings_index, embeddings_size = readGlove(path_to_glove)


######### read corpus
raw_triplets = read_raw_squad(path_to_data)
triplets = tokenize_squad(raw_triplets, embeddings_index)

# find max length of context, question, answer, respectively
# max_len_c, max_len_q, max_len_a = max_length(triplets)

######### corpus preprocessing
# words that do not appear in embeddings, etc

## find all unique tokens in the data (should be a subset of the number of embeddings)
effective_tokens, effective_num_tokens = count_effective_num_tokens(triplets, embeddings_index)
print('effective number of tokens: ' + str(effective_num_tokens))
print('expected initial loss: ' + str(-np.log(1/float(effective_num_tokens))) + '\n')
# build word2index dictionary and index2word dictionary
word2index, index2word = generate_look_up_table(effective_tokens, effective_num_tokens)


print('reading and preprocessing data complete.')
print('found %s unique tokens in the intersection of corpus and word embeddings.' % effective_num_tokens)
if use_cuda:
    print('GPU ready.')
print('')
print('start training...')
print('')


######### set up model
enc_hidden_size = 256
enc_n_layers = 1
enc_num_directions = 2
dec_hidden_size = 256
dec_n_layers = 1
dec_num_directions = 2
batch_size = 50
learning_rate = 0.001

generator = G(embeddings_size, enc_hidden_size, enc_n_layers, enc_num_directions,
                 embeddings_size, dec_hidden_size, effective_num_tokens, dec_n_layers, dec_num_directions,
                 batch_size)

if use_cuda:
    generator = generator.cuda()

optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

# max_length of generated question
max_length = 100
to_file = True

# open the files
if to_file:
    exp_name = 'G_c_a_sep_pretrain_exp_0831'
    path_to_exp_out = '/home/jack/Documents/QA_QG/exp_results_temp/'
    if not os.path.exists(path_to_exp_out+exp_name):
        os.mkdir(path_to_exp_out+exp_name)
    loss_f = 'loss_temp.txt'
    sample_out_f = 'sample_outputs_temp.txt'
    path_to_loss_f = path_to_exp_out + exp_name + '/' + loss_f
    path_to_sample_out_f = path_to_exp_out + exp_name + '/' + sample_out_f
    loss_f = open(path_to_loss_f,'w+')
    sample_out_f = open(path_to_sample_out_f, 'w+')

trainIters(generator, optimizer, batch_size, embeddings_size,
           embeddings_index, word2index, index2word, max_length, triplets, teacher_forcing_ratio,
           to_file, loss_f, sample_out_f, path_to_exp_out,
           n_iters = 5, print_every=1, plot_every=1)

# save the final model
if to_file:
    torch.save(generator, path_to_exp_out + exp_name +'/generator_temp.pth')


