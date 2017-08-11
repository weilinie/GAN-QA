from __future__ import print_function
from __future__ import division

import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../")
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')

from model_zoo import *
from G_train import *
import numpy as np

use_cuda = torch.cuda.is_available()
teacher_forcing_ratio = 0.5 # default in original code is 0.5


######### set paths
# TODO: to run properly, change the following paths and filenames
# default values for the dataset and the path to the project/dataset
dataset = 'squad'
f_name = 'train-v1.1.json'
path_to_dataset = '/home/jack/Documents/QA_QG/data/'
path_to_data = path_to_dataset + dataset + '/' + f_name
GLOVE_DIR = path_to_dataset + 'glove.6B/'
# path for experiment outputs
# exp_name = 'QG_seq2seq_baseline'
path_to_exp_out = '/home/jack/Documents/QA_QG/exp_results_temp/'
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


print('reading and preprocessing data complete.')
print('found %s unique tokens in the intersection of corpus and word embeddings.' % effective_num_tokens)
if use_cuda:
    print('GPU ready.')
print('')
print('start training...')
print('')


######### set up model
hidden_size1 = 256
hidden_size2 = 256
batch_size = 25
# context encoder
encoder = EncoderRNN(embeddings_size, hidden_size1, batch_size)
# decoder
# input_size, hidden_size, output_size, encoder, n_layers=1, num_directions=1, dropout_p=0.1
attn_decoder = AttnDecoderRNN(embeddings_size, hidden_size2, effective_num_tokens,
                                encoder, n_layers=1, num_directions=1, dropout_p=0.1)

if use_cuda:
    t1 = time.time()
    encoder = encoder.cuda()
    t2 = time.time()
    print('time load encoder: ' + str(t2 - t1))
    attn_decoder = attn_decoder.cuda()
    t3 = time.time()
    print('time load decoder: ' + str(t3 - t2))


# max_length of generated question
max_length = 100


trainIters(encoder, attn_decoder, batch_size, embeddings_size,
           embeddings_index, word2index, index2word, max_length, triplets, teacher_forcing_ratio,
           path_to_loss_f, path_to_sample_out_f, path_to_exp_out,
           n_iters=20000, print_every=100, plot_every=10, learning_rate=0.0005)

# save the final model
torch.save(encoder, path_to_exp_out+'/encoder_temp.pth')
torch.save(attn_decoder, path_to_exp_out+'/decoder_temp.pth')


