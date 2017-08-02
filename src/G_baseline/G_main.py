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
import sys
workspace_path = '/home/jack/Documents/QA_QG/GAN-QA/src/util/'
sys.path.insert(0, workspace_path)
from data_proc import *
from util import *
from G_baseline_model import *
from G_train import *
from G_eval import *


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
f_name = 'train-v1.1.json'
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
    c = post_proc_tokenizer(spacynlp.tokenizer(triple[0]))
    q = post_proc_tokenizer(spacynlp.tokenizer(triple[1]))
    a = post_proc_tokenizer(spacynlp.tokenizer(triple[2]))
    data_tokens += c + q + a
data_tokens = list(set(data_tokens)) # find unique
data_tokens = ['SOS', 'EOS', 'UNK'] + data_tokens

num_tokens = len(data_tokens)
effective_tokens = list(set(data_tokens).intersection(embeddings_index.keys()))
print(effective_tokens[0:20])
effective_num_tokens = len(effective_tokens)


# build word2index dictionary and index2word dictionary
word2index = {}
index2word = {}
for i in range(effective_num_tokens):
    index2word[i] = effective_tokens[i]
    word2index[effective_tokens[i]] = i


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
# context encoder
encoder1 = EncoderRNN(embeddings_size, hidden_size1)
# answer encoder
encoder2 = EncoderRNN(embeddings_size, hidden_size2)
# decoder
attn_model ='cat'
attn_decoder1 = AttnDecoderRNN(attn_model, embeddings_size, hidden_size1, effective_num_tokens, 
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
trainIters(encoder1, encoder2, attn_decoder1, embeddings_index, word2index, index2word, data_tokens, max_length, triplets, teacher_forcing_ratio,
            path_to_loss_f, path_to_sample_out_f, path_to_exp_out,
            50000, print_every=1, plot_every = 1)


