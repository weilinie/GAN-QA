# -*- coding: utf-8 -*-

# NOTE: this is NOT tensorflow. This is PyTorch implementation, standalone of GAN.

"""
question generation model

code adapted from <https://github.com/spro/practical-pytorch>`_

made use of seq2seq learning <http://arxiv.org/abs/1409.3215>
and attention mechanism <https://arxiv.org/abs/1409.0473>

input: a paragraph (aka context), and an answer, both represented by a sequence of tokens
output: a question, represented by a sequence of tokens

"""

#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# requirements
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
from __future__ import unicode_literals, print_function, division
import time
from io import open
import unicodedata
import string
import re
import random

import spacy
from spacy.en import English
spacynlp = English()

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import nltk
import json
import numpy as np
import os

use_cuda = torch.cuda.is_available()





#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# the model
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#


######################################################################
# The Encoder
# -----------
class EncoderRNN(nn.Module):
	# output is the same dimension as input (dimension defined by externalword embedding model)
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        # self.embeddings_index = embeddings_index

        # self.embedding = nn.Embedding(input_size, input_dim)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden, embeddings_index):
        # input is a word token
        try:
            embedded = Variable(embeddings_index[input].view(1, 1, -1))
        except KeyError:
            embedded = Variable(embeddings_index['UNK'].view(1, 1, -1))
        # embedded = input.view(1,1,-1)
        if use_cuda:
            embedded = embedded.cuda()
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
        n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # self.embeddings_index = embeddings_index

        # self.attn = nn.Linear(self.input_size+self.hidden_size, self.enc_output_len)
        self.attn_combine = nn.Linear(self.input_size+self.hidden_size, self.input_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs, embeddings_index):

        # because the number of input tokens varies, we move the init of attn to here
        # instead of in __init__ function
        attn = nn.Linear(self.input_size+self.hidden_size, encoder_outputs.size()[0])
        if use_cuda:
            attn = attn.cuda()
        
        try:
            embedded = Variable(embeddings_index[input].view(1, 1, -1))
        except KeyError:
            embedded = Variable(embeddings_index['UNK'].view(1, 1, -1))
        # embedded = input.view(1,1,-1)
        if use_cuda:
            embedded = embedded.cuda()
        # embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result





#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# training and evaluation
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#


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

teacher_forcing_ratio = 1 # default in original code is 0.5

# context = input_variable
def train(context_var, ans_var, question_var, embeddings_index, word2index, 
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
    time2 = time.time()
    print('encoder 1 one pass time: ' + str(time2 - time1))
    # answer encoding
    for ei in range(input_length_answer):
        encoder_output_answer, encoder_hidden_answer = encoder2(
            ans_var[ei], encoder_hidden_answer, embeddings_index)
        encoder_outputs_answer[ei] = encoder_output_answer[0][0]
    time3 = time.time()
    print('encoder 2 one pass time: ' + str(time3 - time2))
    # concat the context encoding and answer encoding
    encoder_output = torch.cat((encoder_output_context, encoder_output_answer),1)
    encoder_outputs = torch.cat((encoder_outputs_context, encoder_outputs_answer),0)

    decoder_input = Variable(embeddings_index['SOS'])
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
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
            
            decoder_input = Variable(embeddings_index(question_var[di]))  # Teacher forcing
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs, embeddings_index)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(embeddings_index(ni))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            
            target = Variable(torch.LongTensor([word2index[question_var[di]]]))
            target = target.cuda() if use_cuda else target

            loss += criterion(decoder_output[0], target)
            if ni == EOS_token:
                break
    time4 = time.time()
    print('decoder one pass time: ' + str(time4 - time3))
    loss.backward()
    time5 = time.time()
    print('backprop one pass time: ' + str(time5 - time4))
    encoder_optimizer1.step()
    time6 = time.time()
    print('encoder 1 optimization one pass step time: ' + str(time6 - time5))
    encoder_optimizer2.step()
    time7 = time.time()
    print('encoder 2 optimization one pass step time: ' + str(time7 - time6))
    decoder_optimizer.step()
    time8 = time.time()
    print('decoder optimization one pass step time: ' + str(time8 - time7))
    return loss.data[0] / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


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

def trainIters(encoder1, encoder2, decoder, embeddings_index, word2index, data_tokens,
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
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # start = time.time()
    training_triplets = [variablesFromTriplets(random.choice(triplets), data_tokens)
                        for i in range(n_iters)]
    # end = time.time()
    # print(end - start)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_triple = training_triplets[iter - 1]
        context_var = training_triple[0]
        ans_var = training_triple[2]
        question_var = training_triple[1]
 
        loss = train(context_var, ans_var, question_var, embeddings_index, word2index,
                     encoder1, encoder2, decoder, encoder_optimizer1, encoder_optimizer2, 
                     decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            print('---sample generated question---')
            # sample a triple and print the generated question
            evaluateRandomly(encoder1, encoder2, decoder, triplets, n=1)
            print()

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            # plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            loss_f.write(str(plot_loss_avg))
            loss_f.write('\n')

    # showPlot(plot_losses)


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


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

def evaluate(encoder1, encoder2, decoder, triple):
    triple_var = variablesFromTriplets(triple)
    context_var = triple_var[0]
    ans_var = triple_var[2]
    input_length_context = context_var.size()[0]
    input_length_answer = ans_var.size()[0]
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
    decoder_input = Variable(embeddings_index['SOS'])
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # decoder_hidden = torch.cat(encoder_hidden_context, encoder_hidden_answer)

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, encoder_outputs.size()[0])

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs, embeddings_index)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(index2word[ni])
        
        decoder_input = Variable(embeddings(ni))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder1, encoder2, decoder, triplets, n=1):
    for i in range(n):
        triple = random.choice(triplets)
        print('context   > ', pair[0])
        print('question  > ', pair[1])
        print('answer    > ', pair[2])
        output_words, attentions = evaluate(encoder1, encoder2, decoder, triple)
        output_sentence = ' '.join(output_words)
        print('generated < ', output_sentence)
        print('')





#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# data loading helper functions
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#


######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase
#

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


######################################################################
# read data specific for SQUAD dataset

def readSQuAD(path_to_data):
    # output (context, question, answer) triplets
    print("Reading dataset...")
    triplets = []
    with open(path_to_data) as f:
        train = json.load(f)
        train = train['data']
        for s in range(0, len(train)):
            samples = train[s]['paragraphs']
            for p in range(0, len(samples)):
                context = samples[p]['context']
                # turn from unicode to ascii and lower case everything
                context = normalizeString(context)
                qas = samples[p]['qas']
                for i in range(0, len(qas)):
                # print('current s,p,i are: ' + str(s)+str(p)+str(i))
                    answers = qas[i]['answers']
                    question = qas[i]['question']
                    # turn from unicode to ascii and lower case everything
                    question = normalizeString(question)
                    for a in range(0, len(answers)):
                        ans_text = answers[a]['text']
                        # turn from unicode to ascii and lower case everything
                        ans_text = normalizeString(ans_text)
                        triplets.append((context, question, ans_text))
    return triplets


# helper function for post processing tokenizer 
# outputs a list of strings
def post_proc_tokenizer(tokenized_sentence):
    proc_tokenized_sentence = []
    for t in range(0, len(tokenized_sentence)):
        token = tokenized_sentence[t].string.strip()
        # first check if the string is number or alphabet only
        if token.isdigit() or token.isalpha():
            proc_tokenized_sentence.append(token)
        # sepatate this token into substrings of only words, numbers, or individual symbols
        else:
            index = -1
            for s in range(0, len(token)):
                if s > index:
                    if token[s].isdigit():
                        # print('find digit')
                        for i in range(s,len(token)):
                            if (not token[i].isdigit()):
                                proc_tokenized_sentence.append(token[s:i])
                                index = i-1
                                break
                            elif (token[i].isdigit()) and (i == len(token)-1):
                                proc_tokenized_sentence.append(token[s:i+1])
                                index = i
                                break
                    elif token[s].isalpha():
                        # print('find alphabet')
                        for i in range(s,len(token)):
                            if (not token[i].isalpha()):
                                proc_tokenized_sentence.append(token[s:i])
                                index = i-1
                                break
                            elif (token[i].isalpha()) and (i == len(token)-1):
                                proc_tokenized_sentence.append(token[s:i+1])
                                index = i
                                break
                    else:
                        # print('find symbol')
                        proc_tokenized_sentence.append(token[s])
                        index += 1
                    # print(index)
    return proc_tokenized_sentence
# test
# x = post_proc_tokenizer(spacynlp.tokenizer(u'mid-1960s'))

# turns a sentence into individual tokens
def tokenizeSentence(sentence, data_tokens):
    tokenized_sentence = spacynlp.tokenizer(sentence)
    # # an additional preprocessing step to separate words and non-words when they appear together
    proc_tokenized_sentence = post_proc_tokenizer(tokenized_sentence)
    # print(proc_tokenized_sentence)
    # tokenized_sentence = [token.string.strip() for token in tokenized_sentence]
    # for t in range(0, len(tokenized_sentence)):
    token_num = len(proc_tokenized_sentence)
    # var = torch.FloatTensor(token_num+1, embeddings_size) #add one dimension for EOS
    # var = torch.FloatTensor(token_num+1)
    var = []
    # var[0] = embeddings_index['SOS']
    for t in range(0, token_num):
        # the first if loop only for experimental use to aviod large vocab size
        if proc_tokenized_sentence[t] not in data_tokens:
            var.append('UNK')
        else:
            var.append(proc_tokenized_sentence[t])
        # try:
        #     temp = word2index(proc_tokenized_sentence[t])
        #     var.append()
        # if proc_tokenized_sentence[t] in embeddings_index.keys():
        #     # var[t] = word2index[proc_tokenized_sentence[t]]
        #     var.append(proc_tokenized_sentence[t])
        # else:
        #     # var[t] = word2index['UNK']
        #     var.append('UNK')
    # add end of sentence token to all sentences
    # var[-1] = word2index['EOS']
    var.append('EOS')
    return var


# change these to pytorch variables to prepare as input to the model
# each context, question, answer is a list of indices
def variablesFromTriplets(triple, embeddings_index):
    context = tokenizeSentence(triple[0], embeddings_index)
    answer = tokenizeSentence(triple[2], embeddings_index)
    question = tokenizeSentence(triple[1], embeddings_index)
    return (context, question, answer)





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
path_to_exp_out = '/home/jack/Documents/QA_QG/GAN-general/exp_results/' + exp_name
loss_f = 'loss.txt'
sample_out_f = 'sample_outputs.txt'
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
    c = [token.string.strip() for token in spacynlp.tokenizer(triple[0])]
    q = [token.string.strip() for token in spacynlp.tokenizer(triple[1])]
    a = [token.string.strip() for token in spacynlp.tokenizer(triple[2])]
    data_tokens += c + q + a
data_tokens = list(set(data_tokens)) # find unique
data_tokens = ['SOS', 'EOS', 'UNK'] + data_tokens
print(data_tokens[0:20])
# experimental usage only
data_tokens = data_tokens[0:10000]

num_tokens = len(data_tokens)
# generate some index
# token_indices = random.sample(range(0, len(data_tokens)), 20)
# # debugging purpose
# token_subset = [data_tokens[i] for i in token_indices]
# print('original tokens: ' + str(token_subset))
# # extra preprocessing step to replace all tokens in data_tokens 
# # that does not appear in embeddings_index to 'UNK'
# # OOV_indices = [i for i, e in enumerate(data_tokens) if e not in set(embeddings_index.keys())] # indices of out of vocabulary words in data_tokens
# for i in OOV_indices:
#     data_tokens[i] = 'UNK'
# # debugging: randomly sample 20 tokens from data_tokens. shouldn't be all UNK
# token_subset = [data_tokens[i] for i in token_indices]
# print('modified tokens: ' + str(token_subset))

# build word2index dictionary and index2word dictionary
word2index = {}
index2word = {}
for i in range(0, len(data_tokens)):
    index2word[i] = data_tokens[i]
    word2index[data_tokens[i]] = i

print('reading and preprocessing data complete.')
print('found %s unique tokens in corpus.' % len(data_tokens))
if use_cuda:
    print('GPU ready.')
print('')
print('start training...')
print('')


######### set up model
hidden_size1 = 64
hidden_size2 = 64
# context encoder
encoder1 = EncoderRNN(embeddings_size, hidden_size1)
# answer encoder
encoder2 = EncoderRNN(embeddings_size, hidden_size2)
# decoder
attn_decoder1 = AttnDecoderRNN(embeddings_size, hidden_size1, num_tokens, 
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


######### start training
trainIters(encoder1, encoder2, attn_decoder1, embeddings_index, word2index, data_tokens,
            path_to_loss_f, path_to_sample_out_f, path_to_exp_out,
            75000, print_every=1)

# save the final model
torch.save(encoder1, path_to_exp_out+'/encoder1.pth')
torch.save(encoder2, path_to_exp_out+'/encoder2.pth')
torch.save(decoder, path_to_exp_out+'/decoder.pth')

######################################################################
#

evaluateRandomly(encoder1, encoder2, attn_decoder1)


######################################################################
# Visualizing Attention
# ---------------------
#
# A useful property of the attention mechanism is its highly interpretable
# outputs. Because it is used to weight specific encoder outputs of the
# input sequence, we can imagine looking where the network is focused most
# at each time step.
#
# You could simply run ``plt.matshow(attentions)`` to see attention output
# displayed as a matrix, with the columns being input steps and rows being
# output steps:
#

output_words, attentions = evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())


######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("elle a cinq ans de moins que moi .")

evaluateAndShowAttention("elle est trop petit .")

evaluateAndShowAttention("je ne crains pas de mourir .")

evaluateAndShowAttention("c est un jeune directeur plein de talent .")

