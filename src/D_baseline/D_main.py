
from __future__ import print_function
from __future__ import division

import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../"))
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
from data_proc import *

from D_model import *
from D_train import *
from D_eval import *
import numpy as np

from torch import optim

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
path_to_exp_out = os.path.abspath(__file__ + '/../../../../') + '/exp_results_D_temp/'
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
enc_hidden_size = 256
enc_n_layers = 1
num_directions = 1
mlp_hidden_size = 64
mlp_output_size = 1
num_attn_weights = 1 # 1000
use_attn = True
batch_size = 100
enc_lr = 0.01
mlp_lr = 0.01
learning_rate = 0.001
discriminator = D(embeddings_size, enc_hidden_size, enc_n_layers, num_directions,
                  mlp_hidden_size, num_attn_weights, mlp_output_size, use_attn,
                  batch_size)
if use_cuda:
    discriminator = discriminator.cuda()

criterion = nn.BCELoss()
optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)


######### start training
to_file = False
train(discriminator, criterion, optimizer, batch_size, embeddings_size,
           embeddings_index, word2index, index2word, triplets,
           to_file, path_to_loss_f, path_to_sample_out_f, path_to_exp_out,
           n_iters=3000, print_every=100, plot_every=1)


# save the final model
# if to_file:
#     torch.save(encoder, path_to_exp_out+'/encoder.pth')
#     torch.save(mlp, path_to_exp_out+'/mlp.pth')





