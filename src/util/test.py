# various test cases

# load model
import sys, os
from data_proc import *
from util import *
sys.path.append(os.path.abspath(__file__ + "/../../"))
sys.path.append(os.path.abspath(__file__ + "/../../") + '/D_baseline')
sys.path.append(os.path.abspath(__file__ + "/../../") + '/G_baseline')
from G_model import *
from model_zoo import *
from G_eval import *
import torch

global use_cuda
use_cuda = torch.cuda.is_available()

######################################################################
######################################################################
# test for various util functions
# uncomment this for much of the later unit tests in this file
######### set paths
# TODO: to run properly, change the following paths and filenames
# default values for the dataset and the path to the project/dataset
dataset = 'squad'
f_name = 'dev-v1.1.json'
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

# # test of windowed triplets
# window_size = 10
# test_idx = 250
# windowed_c_triplets = get_windowed_ans(raw_triplets[test_idx:test_idx+2], window_size)
# print(raw_triplets[test_idx][0])
# print(raw_triplets[test_idx][2])
# print(windowed_c_triplets[0][0])

# test of selecting the sentence containing answer from context
test_idx = 185
sent_c_triplets = get_ans_sentence(raw_triplets[test_idx:test_idx+2])
print(raw_triplets[test_idx][0])
print(raw_triplets[test_idx][2])
print('ans start idx: %d' % raw_triplets[test_idx][3])
print('ans end idx:   %d' % raw_triplets[test_idx][4])
print(sent_c_triplets[0][0])


# triplets = tokenize_squad(raw_triplets, embeddings_index)

# # find max length of context, question, answer, respectively
# max_len_c, max_len_q, max_len_a = max_length(triplets)

# effective_tokens, effective_num_tokens = count_effective_num_tokens(triplets, embeddings_index)
# print('effective number of tokens: ' + str(effective_num_tokens))
# print('expected initial loss: ' + str(-np.log(1/float(effective_num_tokens))) + '\n')
# # build word2index dictionary and index2word dictionary
# word2index, index2word = generate_look_up_table(effective_tokens, effective_num_tokens)


######################################################################
######################################################################
# # test case of get_random_batch and prepare_batch_var functions in data_proc.py
# # (uncomment code below to test)
# # test and time
# # to run this test, you need to have these things ready:
# # 1) triplet processed by tokenize_squad,
# # 2) embeddings_index
# # 3) a mini batch processed by get_random_batch
# batch_size = 500
# start = time.time()
# batch, seq_lens, fake_batch, fake_seq_lens = get_random_batch(triplets, batch_size, with_fake=True)
#
# temp, temp_orig, seq_lens_cqa = prepare_batch_var(batch, seq_lens, fake_batch, fake_seq_lens, batch_size, word2index, embeddings_index, embeddings_size,
#                                                   mode = ['word', 'index'], concat_opt='cqa', with_fake=True)
# end = time.time()
# print('time elapsed: ' + str(end-start))
# # the following check if the batched data matches with the original data
# batch_idx = random.choice(range(batch_size))
# print(batch_idx)
#
# print('context  > ', ' '.join(temp_orig[0][batch_idx]))
# print('question > ', ' '.join(temp_orig[1][batch_idx]))
# print('answer   > ', ' '.join(temp_orig[2][batch_idx]))
#
# idx = batch[0].index(temp_orig[0][batch_idx])
# print('context  > ', ' '.join(batch[0][idx]))
# print('question > ', ' '.join(batch[1][idx]))
# print('answer   > ', ' '.join(batch[2][idx]))

# seq_idx = random.choice(range(min(seq_lens[0])))
# print(seq_idx)
# word1 = embeddings_index[batch[0][seq_lens[0].index(heapq.nlargest(batch_idx, seq_lens[0])[-1])][seq_idx]]
# word2 = temp[0][seq_idx, batch_idx,]
# set(word1) == set(word2.data.cpu())


######################################################################
######################################################################
# # test case to load the G model and sample from G
# teacher_forcing_ratio = 0.5 # default in original code is 0.5

# # param for G
# enc_hidden_size = 256
# enc_n_layers = 1
# enc_num_directions = 1
# dec_hidden_size = 256
# dec_n_layers = 1
# dec_num_directions = 1
# batch_size = 5
# learning_rate = 0.0005

# generator = G(embeddings_size, enc_hidden_size, enc_n_layers, enc_num_directions,
#                  embeddings_size, dec_hidden_size, effective_num_tokens, dec_n_layers, dec_num_directions,
#                  batch_size)
# if use_cuda:
#     generator = generator.cuda()

# # prepare G input
# training_batch, seq_lens = get_random_batch(triplets, batch_size)
# ca = training_batch[0][0] + training_batch[2][0]
# # sample from G
# max_len = 100
# sample_q = G_sampler(generator, ca, embeddings_index, embeddings_size, word2index, index2word, max_len)
# print(' '.join(sample_q))