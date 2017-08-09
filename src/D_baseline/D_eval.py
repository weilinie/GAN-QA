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

from ..util.data_proc import *

use_cuda = torch.cuda.is_available()

######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder1, encoder2, MLP, triple):
    triple_var = variablesFromTriplets(triple)
    context_var = triple_var[0]
    question_var = triple_var[1]
    ans_start_idx = triple_var[3]
    ans_end_idx = triple_var[4]
    num_char_context = len(triple[0])
    input_length_context = context_var.size()[0]
    input_length_question = question_var.size()[0]
    encoder_hidden_context = encoder1.initHidden()
    encoder_hidden_question = encoder2.initHidden()


    encoder_hiddens_context = Variable(torch.zeros(input_length_context, encoder1.hidden_size))
    encoder_hiddens_context = encoder_hiddens_context.cuda() if use_cuda else encoder_hiddens_context
    encoder_hiddens_question = Variable(torch.zeros(input_length_question, encoder2.hidden_size))
    encoder_hiddens_question = encoder_hiddens_question.cuda() if use_cuda else encoder_hiddens_question
   
    for ei in range(input_length_context):
        encoder_output_context, encoder_hidden_context = encoder1(context_var[ei],
                                                 encoder_hidden_context, embeddings_index)
        encoder_hiddens_context[ei] = encoder_hiddens_context[ei] + encoder_hidden_context[0][0]

    for ei in range(input_length_answer):
        encoder_output_answer, encoder_hidden_answer = encoder2(ans_var[ei],
                                                 encoder_hidden_answer, embeddings_index)
        encoder_hiddens_answer[ei] = encoder_hiddens_answer[ei] + encoder_hidden_answer[0][0]

    output = MLP( torch.cat((encoder_hiddens_context, encoder_hiddens_question),0) )
    output = output[0:num_char_context]

    pred_ans_start_idx = sfmx1(output)
    pred_ans_start_idx = pred_ans_start_idx.data.max(1)[1]
    pred_ans_end_idx = sfmx2(output)
    pred_ans_end_idx = pred_ans_end_idx.data.max(1)[1]

    pred_ans = triple[0][pred_ans_start_idx:pred_ans_end_idx]

    return pred_ans, (pred_ans_start_idx, pred_ans_end_idx)


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder1, encoder2, MLP, triplets, n=1):
    for i in range(n):
        triple = random.choice(triplets)
        print('context   > ', triple[0])
        print('question  > ', triple[1])
        print('answer    > ', triple[2])
        pred_ans, pred_idx = evaluate(encoder1, encoder2, MLP, triple)
        print('predicted answer < ', pred_ans)
        print('')
