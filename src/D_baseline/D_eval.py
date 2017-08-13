
import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../"))
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')

from data_proc import *
from D_model import *

import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


def evaluate(discriminator, triplets,
             word2index, embeddings_index, embeddings_size,
             eval_batch_size=10):
    
    # prepare batch
    training_batch, seq_lens, fake_training_batch, fake_seq_lens = get_random_batch(triplets, eval_batch_size, with_fake=True)
    # concat the context_ans batch with the question batch
    # each element in the training batch is context + question + answer
    training_batch, _, seq_lens = prepare_batch_var(training_batch, seq_lens, fake_training_batch, fake_seq_lens,
                                                    eval_batch_size, word2index, embeddings_index, embeddings_size,
                                                    mode = ['word'], concat_opt='cqa', with_fake=True)

    train_input = Variable(training_batch[0].cuda()) if use_cuda else Variable(
        training_batch[0])  # embeddings vectors, size = [seq len x batch size x embedding dim]
    true_labels = Variable(torch.FloatTensor(training_batch[-1]).cuda()) if use_cuda else Variable(
        torch.FloatTensor(training_batch[-1]))

    # pass through discriminator model
    outputs = discriminator.forward(train_input, true_labels, seq_lens[0])

    # get label predictions from model & compare the number of correct predictions
    pred_labels = torch.zeros(outputs.size())
    num_correct_pred = 0
    for i in range(outputs.size(0)):
        pred_labels[i] = 0 if outputs.data[i][0] <= 0.5 else 1
        if pred_labels[i][0] == true_labels[i].data[0]:
            num_correct_pred += 1

    print('percentage of correct predictions (True/False): ' + 
            str(float(num_correct_pred)/float(outputs.size(0))*100) + '%.\n')



    
