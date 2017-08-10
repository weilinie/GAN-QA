
import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
# print(os.path.abspath(__file__ + '/../../../../')+'/util')
sys.path.append(os.path.abspath(__file__ + "/../../") + '/G_baseline_batch')

from data_proc import *
from model_zoo import *

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import time

use_cuda = torch.cuda.is_available()

# # load the saved models
# path_to_model = os.path.abspath(__file__ + '/../../../../') + '/exp_results_temp/'
# encoder = torch.load(path_to_model + 'encoder.pth')
# mlp = torch.load(path_to_model + 'mlp.pth')
#
# # load data
# # default values for the dataset and the path to the project/dataset
# dataset = 'squad'
# f_name = 'dev-v1.1.json'
# path_to_dataset = os.path.abspath(__file__ + '/../../../../') + '/data/'
# path_to_data = path_to_dataset + dataset + '/' + f_name
# GLOVE_DIR = path_to_dataset + 'glove.6B/'
# # path for experiment outputs
# # exp_name = 'QG_seq2seq_baseline'
# path_to_exp_out = os.path.abspath(__file__ + '/../../../../') + '/exp_results_temp/'
# loss_f = 'loss_temp.txt'
# sample_out_f = 'sample_outputs_temp.txt'
# path_to_loss_f = path_to_exp_out + '/' + loss_f
# path_to_sample_out_f = path_to_exp_out + '/' + sample_out_f
#
#
# ######### first load the pretrained word embeddings
# path_to_glove = os.path.join(GLOVE_DIR, 'glove.6B.50d.txt')
# embeddings_index, embeddings_size = readGlove(path_to_glove)
#
#
# ######### read corpus
# raw_triplets = read_raw_squad(path_to_data)
# triplets = tokenize_squad(raw_triplets, embeddings_index)
#
# # find max length of context, question, answer, respectively
# max_len_c, max_len_q, max_len_a = max_length(triplets)
#
# ######### corpus preprocessing
# # words that do not appear in embeddings, etc
#
# ## find all unique tokens in the data (should be a subset of the number of embeddings)
# effective_tokens, effective_num_tokens = count_effective_num_tokens(triplets, embeddings_index)
# print('effective number of tokens: ' + str(effective_num_tokens))
# print('expected initial loss: ' + str(-np.log(1/float(effective_num_tokens))) + '\n')
# # build word2index dictionary and index2word dictionary
# word2index, index2word = generate_look_up_table(effective_tokens, effective_num_tokens)

######################################################################
# Evaluation
# ==========
#
# get a batch of real and fake questions, record their labels,
# and check the model output
# predicted T/F: since we used sigmoid in D_train, we will simply 
# assign the example to be True if the output from sigmoid > 0.5, 
# otherwise False
#

def evaluate(encoder, mlp, triplets,
             word2index, embeddings_index, embeddings_size,
             eval_batch_size=10):
    
    # read a batch of true and fake data
    # size of true data = size of fake data (different setup compared to data_proc.py)
    training_batch, seq_lens, fake_training_batch, fake_seq_lens = get_random_batch(triplets, eval_batch_size, with_fake=True)
    # c_a_q = list(training_batch[0])
    # concat the context_ans batch with the question batch
    # each element in the training batch is context + question + answer
    training_batch, _, seq_lens = prepare_batch_var(training_batch, seq_lens, fake_training_batch, fake_seq_lens,
                                                    eval_batch_size, word2index, embeddings_index, embeddings_size,
                                                    mode = ['word', 'index'], concat_opt='cqa', with_fake=True)

    train_input = Variable(training_batch[0].cuda()) if use_cuda else Variable(
        training_batch[0])  # embeddings vectors, size = [seq len x batch size x embedding dim]
    true_labels = Variable(torch.FloatTensor(training_batch[-1]).cuda()) if use_cuda else Variable(
        torch.FloatTensor(training_batch[-1]))

    # pass through discriminator model
    encoder_hiddens, encoder_hidden = encoder(train_input, seq_lens[0], None)
    outputs = mlp(encoder_hiddens)


    # get label predictions from model & compare the number of correct predictions
    pred_labels = torch.zeros(outputs.size())
    num_correct_pred = 0
    for i in range(outputs.size(0)):
        pred_labels[i] = 0 if outputs.data[i][0] <= 0.5 else 1
        if pred_labels[i][0] == true_labels[i].data[0]:
            num_correct_pred += 1


    # print(outputs.data[0][0])
    # print(outputs.data[0][0] > 0.5)
    # print(outputs.data.t())
    # print(pred_labels.t())
    # print(true_labels.data.unsqueeze(1).t())
    print('percentage of correct predictions (True/False): ' + 
            str(float(num_correct_pred)/float(outputs.size(0))*100) + '%.\n')



# evaluate(encoder, mlp, triplets, word2index, embeddings_index, embeddings_size, eval_batch_size=10)
