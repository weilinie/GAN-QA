# -*- coding: utf-8 -*-

# NOTE: this is NOT tensorflow. This is PyTorch implementation, standalone of GAN.

"""
question generation model baseline

code adapted from <https://github.com/spro/practical-pytorch>`_

made use of seq2seq learning <http://arxiv.org/abs/1409.3215>
and attention mechanism <https://arxiv.org/abs/1409.0473>

input: a paragraph (aka context), and an answer, both represented by a sequence of tokens
output: a question, represented by a sequence of tokens

"""

from __future__ import print_function
from __future__ import division

from ..util.data_proc import *
from model_zoo import *
from G_train_batch import *
from G_eval_batch import *


use_cuda = torch.cuda.is_available()
teacher_forcing_ratio = 0.5 # default in original code is 0.5

#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# evaluation script
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#


######### set paths
# default values for the dataset and the path to the project/dataset
dataset = 'squad'
f_name = 'dev-v1.1.json'
path_to_dataset = '/home/jack/Documents/QA_QG/data/'
path_to_data = path_to_dataset + dataset + '/' + f_name
GLOVE_DIR = path_to_dataset + 'glove.6B/'
# path for experiment outputs
exp_name = 'QG_seq2seq_baseline'
path_to_exp_out = '/home/jack/Documents/QA_QG/exp_results/' + exp_name
loss_f = 'loss_temp.txt'
sample_out_f = 'sample_outputs_temp.txt'
path_to_loss_f = path_to_exp_out + '/' + loss_f
path_to_sample_out_f = path_to_exp_out + '/' + sample_out_f


######### first load the pretrained word embeddings
path_to_glove = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')
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
batch_size = 200
# context encoder
encoder1 = EncoderRNN(embeddings_size, hidden_size1)
# answer encoder
encoder2 = EncoderRNN(embeddings_size, hidden_size2)
# decoder
attn_decoder1 = AttnDecoderRNN(embeddings_size, hidden_size1, effective_num_tokens, 
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


# max_length of generated question
max_length = 100

######### start training
trainIters(encoder1, encoder2, attn_decoder1, embeddings_index, 
            word2index, index2word, max_length, triplets, teacher_forcing_ratio,
            path_to_loss_f, path_to_sample_out_f, path_to_exp_out,
            1, print_every=1, plot_every = 1)

# save the final model
torch.save(encoder1, path_to_exp_out+'/encoder1_temp.pth')
torch.save(encoder2, path_to_exp_out+'/encoder2_temp.pth')
torch.save(attn_decoder1, path_to_exp_out+'/decoder_temp.pth')


