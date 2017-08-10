import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import time

import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
# print(os.path.abspath(__file__ + '/../../../../')+'/util')
sys.path.append(os.path.abspath(__file__ + "/../../") + '/G_baseline_batch')

from data_proc import *

use_cuda = torch.cuda.is_available()

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

def evaluate(encoder, MLP, triplets, eval_batch_size):
    
    # read a batch of true and fake data
    # size of true data = size of fake data (different setup compared to data_proc.py)
    training_batch, seq_lens, fake_training_batch, fake_seq_lens = get_random_batch(triplets, eval_batch_size, with_fake=True)
    # c_a_q = list(training_batch[0])
    # concat the context_ans batch with the question batch
    # each element in the training batch is context + question + answer
    training_batch, _, seq_lens = prepare_batch_var(training_batch, seq_lens, fake_training_batch, fake_seq_lens,
                                                    batch_size, word2index, embeddings_index, embeddings_size,
                                                    mode = ['word', 'index'], concat_opt='cqa', with_fake=True)
    
    train_input = training_batch[0] # embeddings vectors, size = [seq len x batch size x embedding dim]
    true_labels = training_batch[-1]

    # pass through discriminator model
    encoder_hiddens, encoder_hidden = encoder(train_batch, seq_lens[0], None)
    outputs = F.sigmoid(mlp(encoder_hiddens))

    # get label predictions from model & compare the number of correct predictions
    pred_labels = torch.zeros(outputs.size())
    num_correct_pred = 0
    for i in range(output.size(0)):
        pred_labels[i] = 0 if outputs[i] <= 0 else pred_labels[i] = 1
        if pred_labels[i] == true_labels[i]:
            num_correct_pred += 1

    print('percentage of correct predictions (True/False): ' + 
            str(float(num_correct_pred)/float(outputs.size(0))) + '%.\n')



