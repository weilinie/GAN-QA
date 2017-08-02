#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# training and evaluation
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
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
from util import *

use_cuda = torch.cuda.is_available()

######################################################################
# Training the Model
# ------------------
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability 

# context = input_variable
def train(context_var, ans_var, question_var, embeddings_index, word2index, index2word, teacher_forcing_ratio,
    encoder1, encoder2, decoder, encoder_optimizer1, encoder_optimizer2, 
    decoder_optimizer, criterion):
    encoder_hidden_context = encoder1.initHidden()
    encoder_hidden_answer = encoder2.initHidden()
    decoder_hidden = decoder.initHidden()

    encoder_optimizer1.zero_grad()
    encoder_optimizer2.zero_grad()
    decoder_optimizer.zero_grad()

    input_length_context = len(context_var)
    input_length_answer = len(ans_var)
    target_length = len(question_var)
    
    encoder_outputs_context = Variable(torch.zeros(input_length_context, encoder1.hidden_size))
    encoder_outputs_context = encoder_outputs_context.cuda() if use_cuda else encoder_outputs_context

    encoder_outputs_answer = Variable(torch.zeros(input_length_answer, encoder2.hidden_size))
    encoder_outputs_answer = encoder_outputs_answer.cuda() if use_cuda else encoder_outputs_answer
   
    loss = 0

    # context encoding
    time1 = time.time()
    for ei in range(input_length_context):
    	encoder_output_context, encoder_hidden_context = encoder1(
        	context_var[ei], encoder_hidden_context, embeddings_index)
    	encoder_outputs_context[ei] = encoder_output_context[0][0]
    # time2 = time.time()
    # print('encoder 1 one pass time: ' + str(time2 - time1))
    # answer encoding
    for ei in range(input_length_answer):
        encoder_output_answer, encoder_hidden_answer = encoder2(
            ans_var[ei], encoder_hidden_answer, embeddings_index)
        encoder_outputs_answer[ei] = encoder_output_answer[0][0]
    # time3 = time.time()
    # print('encoder 2 one pass time: ' + str(time3 - time2))
    # concat the context encoding and answer encoding
    encoder_output = torch.cat((encoder_output_context, encoder_output_answer),1)
    encoder_outputs = torch.cat((encoder_outputs_context, encoder_outputs_answer),0)

    decoder_input = 'SOS' # Variable(embeddings_index['SOS'])
    # decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    # decoder_hidden = torch.cat(encoder_hidden_context, encoder_hidden_answer)

    # #debug
    # print(decoder_input.is_cuda)
    # print(decoder_hidden.is_cuda)
    # print(encoder_output.is_cuda)
    # print(encoder_outputs.is_cuda)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs, embeddings_index)

            target = Variable(torch.LongTensor([word2index[question_var[di]]]))
            target = target.cuda() if use_cuda else target

            loss += criterion(decoder_output[0], target)
            
            decoder_input = question_var[di] # Variable(embeddings_index[question_var[di]])  # Teacher forcing
            # decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs, embeddings_index)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            # print(ni)
            # print(type(ni))
            
            decoder_input = index2word[di] # Variable(embeddings_index[index2word[ni]])
            # decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            
            target = Variable(torch.LongTensor([word2index[question_var[di]]]))
            target = target.cuda() if use_cuda else target

            loss += criterion(decoder_output[0], target)
            if ni == word2index['EOS']:
                break
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

def trainIters(encoder1, encoder2, decoder, 
    embeddings_index, word2index, index2word, data_tokens, max_length, triplets, teacher_forcing_ratio,
    path_to_loss_f, path_to_sample_out_f, path_to_exp_out,
    n_iters, print_every=10, plot_every=100, learning_rate=0.01):

    begin_time = time.time()

    # open the files
    loss_f = open(path_to_loss_f,'w+') 
    sample_out_f = open(path_to_sample_out_f, 'w+')

    # plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer1 = optim.SGD(encoder1.parameters(), lr=learning_rate)
    encoder_optimizer2 = optim.SGD(encoder2.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # start = time.time()
    # training_triplets = [variablesFromTriplets(random.choice(triplets), data_tokens)
    #                     for i in range(n_iters)]
    # end = time.time()
    # print('time spent prepare data: ' + str(end - start))
    criterion = nn.NLLLoss()

    print()

    for iter in range(1, n_iters + 1):
        # training_triple = training_triplets[iter - 1]
        training_triple = variablesFromTriplets(random.choice(triplets), embeddings_index)
        context_var = training_triple[0]
        ans_var = training_triple[2]
        question_var = training_triple[1]
        
        start = time.time()
        loss = train(context_var, ans_var, question_var, embeddings_index, word2index, index2word, teacher_forcing_ratio,
                     encoder1, encoder2, decoder, encoder_optimizer1, encoder_optimizer2, 
                     decoder_optimizer, criterion)
        end = time.time()


        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(begin_time, iter / float(n_iters)),
                                         iter, iter / n_iters * 100, print_loss_avg))
            print('time for one training iteration: ' + str(end - start))
            print('---sample generated question---')
            # sample a triple and print the generated question
            evaluateRandomly(encoder1, encoder2, decoder, triplets, embeddings_index, word2index, index2word, max_length, n=1)
            print('-------------------------------')
            print('-------------------------------')
            print()

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            # plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            loss_f.write(unicode(plot_loss_avg))
            loss_f.write(unicode('\n'))

    # showPlot(plot_losses)
    loss_f.close()





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
# max_length constrains the maximum length of the generated question
def evaluate(encoder1, encoder2, decoder, triple, embeddings_index, word2index, index2word, max_length):
    triple_var = variablesFromTriplets(triple, embeddings_index)
    context_var = triple_var[0]
    ans_var = triple_var[2]
    input_length_context = len(context_var)
    input_length_answer = len(ans_var)
    encoder_hidden_context = encoder1.initHidden()
    encoder_hidden_answer = encoder2.initHidden()
    decoder_hidden = decoder.initHidden()


    encoder_outputs_context = Variable(torch.zeros(input_length_context, encoder1.hidden_size))
    encoder_outputs_context = encoder_outputs_context.cuda() if use_cuda else encoder_outputs_context
    encoder_outputs_answer = Variable(torch.zeros(input_length_answer, encoder2.hidden_size))
    encoder_outputs_answer = encoder_outputs_answer.cuda() if use_cuda else encoder_outputs_answer
   
    for ei in range(input_length_context):
        encoder_output_context, encoder_hidden_context = encoder1(context_var[ei],
                                                 encoder_hidden_context, embeddings_index)
        encoder_outputs_context[ei] = encoder_outputs_context[ei] + encoder_output_context[0][0]

    for ei in range(input_length_answer):
        encoder_output_answer, encoder_hidden_answer = encoder2(ans_var[ei],
                                                 encoder_hidden_answer, embeddings_index)
        encoder_outputs_answer[ei] = encoder_outputs_answer[ei] + encoder_output_answer[0][0]

    encoder_output = torch.cat((encoder_output_context, encoder_output_answer))
    encoder_outputs = torch.cat((encoder_outputs_context, encoder_outputs_answer))

    # decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = 'SOS'  # Variable(embeddings_index['SOS'])
    # decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # decoder_hidden = torch.cat(encoder_hidden_context, encoder_hidden_answer)

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, encoder_outputs.size()[0])

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs, embeddings_index)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        # print(ni)
        # print(type(ni))

        if ni == word2index['EOS']:
            decoded_words.append('EOS')
            break
        else:
            decoded_words.append(index2word[ni])
        
        decoder_input = index2word[ni] # Variable(embeddings_index[index2word[ni]])
        # decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder1, encoder2, decoder, triplets, embeddings_index, word2index, index2word, max_length, n=1):
    for i in range(n):
        triple = random.choice(triplets)
        print('context   > ', triple[0])
        print('question  > ', triple[1])
        print('answer    > ', triple[2])
        output_words, attentions = evaluate(encoder1, encoder2, decoder, triple, embeddings_index, word2index, index2word, max_length)
        output_sentence = ' '.join(output_words)
        print('generated < ', output_sentence)
        print('')