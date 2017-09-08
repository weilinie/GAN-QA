from __future__ import print_function
from __future__ import division

import sys, os
# sys.path.append(os.path.abspath(__file__ + "/../../"))
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
# sys.path.append(os.path.abspath(__file__ + "/../../") + '/G_c_a_sep')
# sys.path.append(os.path.abspath(__file__ + "/../../") + '/D_baseline')

from data_proc import *
# from G_train import *
# from G_c_a_sep import *
from GAN_model import *
import numpy as np

from torch import optim

global use_cuda
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
path_to_glove = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')
embeddings_index, embeddings_size = readGlove(path_to_glove)


import pickle
load_path = '/home/jack/Documents/QA_QG/data/processed/'
# triplets = pickle.load(open(load_path+'triplets.txt', 'rb'))
sent_c_triplets = pickle.load(open(load_path+'sent_c_triplets.txt', 'rb'))
# windowed_c_triplets_10 = pickle.load(open(load_path+'windowed_c_triplets_10.txt', 'rb'))
triplets = sent_c_triplets
# ######### read corpus
# raw_triplets = read_raw_squad(path_to_data)
# triplets = tokenize_squad(raw_triplets, embeddings_index)

# # find max length of context, question, answer, respectively
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
G_enc_input_size = embeddings_size
G_enc_hidden_size = 256
G_enc_n_layers = 1
G_enc_num_directions = 1
G_dec_input_size = embeddings_size
G_dec_hidden_size = 256
G_output_size = effective_num_tokens
G_dec_n_layers = 1
G_dec_num_directions = 1
D_enc_input_size = embeddings_size
D_enc_hidden_size = 256
D_enc_n_layers = 1
D_num_directions = 1
D_mlp_hidden_size = 64
D_num_attn_weights = 1
D_mlp_output_size = 1
use_attn = True
batch_size = 5

G_path = '/home/jack/Documents/QA_QG/exp_results_temp/G_c_a_sep_pretrain_exp_0902(2)/generator_temp.pth'

vanilla_gan = GAN_model(G_enc_input_size, G_enc_hidden_size, G_enc_n_layers, G_enc_num_directions,
                        G_dec_input_size, G_dec_hidden_size, G_output_size, G_dec_n_layers, G_dec_num_directions,
                        D_enc_input_size, D_enc_hidden_size, D_enc_n_layers, D_num_directions,
                        D_mlp_hidden_size, D_num_attn_weights, D_mlp_output_size,
                        use_attn, batch_size, G_path=G_path, pretrain=True)
if use_cuda:
    vanilla_gan = vanilla_gan.cuda()

learning_rate = 1e-3
d_optimizer = optim.Adam(vanilla_gan.D.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(vanilla_gan.G.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# max_length of generated question
max_len = 100
to_file = True
print_every = 50
plot_every = 500
checkpoint_every = 2000
n_iters = 10000
d_steps = 1
g_steps = 5

# open the files
exp_name = 'GAN_0907'
path_to_exp = '/home/jack/Documents/QA_QG/exp_results_temp/'
path_to_exp_out = path_to_exp + exp_name
if to_file:
    if not os.path.exists(path_to_exp_out):
        os.mkdir(path_to_exp_out)
    loss_f = 'loss_temp.txt'
    sample_out_f = 'sample_outputs_temp.txt'
    path_to_loss_f = path_to_exp_out + '/' + loss_f
    path_to_sample_out_f = path_to_exp_out + '/' + sample_out_f
    loss_f = open(path_to_loss_f,'w+')
    sample_out_f = open(path_to_sample_out_f, 'w+')
# else:
#     loss_f = None
#     sample_out_f = None
#     path_to_exp_out = None

# load a pre-trained model
model_fname = 'checkpoint_iter_1.pth.tar'
path_to_model = path_to_exp_out + '/' + model_fname
checkpoint = torch.load(path_to_model)
vanilla_gan.D.load_state_dict(checkpoint['d_state_dict'])
vanilla_gan.G.load_state_dict(checkpoint['g_state_dict'])
d_optimizer.load_state_dict(checkpoint['d_optimizer'])
g_optimizer.load_state_dict(checkpoint['g_optimizer'])

# train
vanilla_gan.train(triplets, n_iters, d_steps, d_optimizer, g_steps, g_optimizer, batch_size, max_len,
                  criterion, word2index, index2word, embeddings_index, embeddings_size, print_every, plot_every, checkpoint_every,
                  to_file=to_file, loss_f=loss_f, sample_out_f=sample_out_f, path_to_exp_out=path_to_exp_out)

if to_file:
    loss_f.close()
    sample_out_f.close()
    torch.save(vanilla_gan, path_to_exp_out + exp_name + '/GAN_model.pth.tar')