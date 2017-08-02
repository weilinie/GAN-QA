import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import time

import sys
workspace_path = '/home/jack/Documents/QA_QG/GAN-QA/src/util/'
sys.path.insert(0, workspace_path)
from data_proc import *

use_cuda = torch.cuda.is_available()

#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# training and evaluation
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#


######################################################################
# Training the Model
# ------------------


# context = input_variable
def train(context_var, question_var, train_triple_raw, 
    embeddings_index, word2index, ans_start_idx, ans_end_idx,
    encoder1, encoder2, MLP, encoder_optimizer1, encoder_optimizer2, MLP_optimizer, criterion):
    encoder_hidden_context = encoder1.initHidden()
    encoder_hidden_question = encoder2.initHidden()

    encoder_optimizer1.zero_grad()
    encoder_optimizer2.zero_grad()
    MLP_optimizer.zero_grad()

    input_length_context = len(context_var)
    num_char_context = len(train_triple_raw[0]) # which is the untokenized version of the context
    input_length_question = len(qeustion_var)
    
    encoder_hiddens_context = Variable(torch.zeros(input_length_context, encoder1.hidden_size))
    encoder_hiddens_context = encoder_outputs_context.cuda() if use_cuda else encoder_outputs_context

    encoder_hiddens_question = Variable(torch.zeros(input_length_question, encoder2.hidden_size))
    encoder_hiddens_question = encoder_outputs_question.cuda() if use_cuda else encoder_outputs_question
   
    loss = 0

    # context encoding
    time1 = time.time()
    for ei in range(input_length_context):
        encoder_output_context, encoder_hidden_context = encoder1(
            context_var[ei], encoder_hidden_context, embeddings_index)
        encoder_hiddens_context[ei] = encoder_hidden_context[0][0]
    time2 = time.time()
    print('encoder 1 one pass time: ' + str(time2 - time1))
    
    # question encoding
    for ei in range(input_length_question):
        encoder_output_question, encoder_hidden_question = encoder2(
            question_var[ei], encoder_hidden_question, embeddings_index)
        encoder_hiddens_question[ei] = encoder_hidden_question[0][0]
    time3 = time.time()
    print('encoder 2 one pass time: ' + str(time3 - time2))
    
    # concat the context encoding and question encoding
    output = MLP( torch.cat((encoder_hiddens_context, encoder_hiddens_question),0) )
    output = output[0:num_char_context]

    pred_ans_start_idx = sfmx1(output)
    pred_ans_end_idx = sfmx2(output)

    loss += criterion(pred_ans_start_idx, ans_start_idx) + criterion(pred_ans_end_idx, ans_end_idx)

    
    # time4 = time.time()
    # print('decoder one pass time: ' + str(time4 - time3))
    loss.backward()
    # time5 = time.time()
    # print('backprop one pass time: ' + str(time5 - time4))
    encoder_optimizer1.step()
    # time6 = time.time()
    # print('encoder 1 optimization one pass step time: ' + str(time6 - time5))
    encoder_optimizer2.step()
    # time7 = time.time()
    # print('encoder 2 optimization one pass step time: ' + str(time7 - time6))
    decoder_optimizer.step()
    # time8 = time.time()
    # print('decoder optimization one pass step time: ' + str(time8 - time7))
    return loss.data[0] / target_length



######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(encoder1, encoder2, MLP, embeddings_index, word2index, data_tokens, triplets,
    path_to_loss_f, path_to_sample_out_f, path_to_exp_out,
    n_iters, print_every=1000, plot_every=100, learning_rate=0.01):

    # open the files
    loss_f = open(path_to_loss_f,'w+') 
    sample_out_f = open(path_to_sample_out_f, 'w+')

    # plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer1 = optim.SGD(encoder1.parameters(), lr=learning_rate)
    encoder_optimizer2 = optim.SGD(encoder2.parameters(), lr=learning_rate)
    MLP_optimizer = optim.SGD(MLP.parameters(), lr=learning_rate)

    start = time.time()
    
    end = time.time()
    print('time spent prepare data: ' + str(end - start))
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        train_triple_raw = random.choice(triplets)
        training_triple = variablesFromTriplets(train_triple_raw, embeddings_index)
        context_var = training_triple[0]
        ans_var = training_triple[2]
        question_var = training_triple[1]
        ans_start_idx = training_triple[3]
        ans_end_idx = training_triple[4]
        
        loss = train(context_var, question_var, train_triple_raw, 
                        embeddings_index, word2index, ans_start_idx, ans_end_idx,
                        encoder1, encoder2, MLP, encoder_optimizer1, encoder_optimizer2, MLP_optimizer, criterion):

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            print('---sample answer prediction---')
            # sample a triple and print the generated question
            evaluateRandomly(encoder1, encoder2, MLP, triplets, n=1)
            print()

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            # plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            loss_f.write(str(plot_loss_avg))
            loss_f.write('\n')

    # showPlot(plot_losses)

