import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import time

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
# max_length constrains the maximum length of the generated question
def evaluate(encoder, decoder, triple, embeddings_index, word2index, index2word, max_length):
    triple_var = variablesFromTriplets(triple, embeddings_index)
    context_var = triple_var[0]
    ans_var = triple_var[2]
    input_length_context = len(context_var)
    input_length_answer = len(ans_var)
    encoder_hidden_context = encoder1.initHidden()
    encoder_hidden_answer = encoder2.initHidden()
    decoder_hidden = decoder.initHidden()


    encoder_hiddens_context = Variable(torch.zeros(input_length_context, encoder1.hidden_size))
    encoder_hiddens_context = encoder_hiddens_context.cuda() if use_cuda else encoder_hiddens_context
    encoder_hiddens_answer = Variable(torch.zeros(input_length_answer, encoder2.hidden_size))
    encoder_hiddens_answer = encoder_hiddens_answer.cuda() if use_cuda else encoder_hiddens_answer
   
    for ei in range(input_length_context):
        encoder_output_context, encoder_hidden_context = encoder1(context_var[ei],
                                                 encoder_hidden_context, embeddings_index)
        encoder_hiddens_context[ei] = encoder_hiddens_context[ei] + encoder_hidden_context[0][0]

    for ei in range(input_length_answer):
        encoder_output_answer, encoder_hidden_answer = encoder2(ans_var[ei],
                                                 encoder_hidden_answer, embeddings_index)
        encoder_hiddens_answer[ei] = encoder_hiddens_answer[ei] + encoder_hidden_answer[0][0]

    encoder_hiddens = torch.cat((encoder_hiddens_context, encoder_hiddens_answer))

    decoder_input = 'SOS'  # Variable(embeddings_index['SOS'])

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, encoder_hiddens.size()[0])
    if use_cuda:
        decoder_attentions = decoder_attentions.cuda()

    # generate words and store attention values
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_hiddens, embeddings_index)
        decoder_attentions[di,] = decoder_attention.data[0]
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        # print(ni)
        # print(type(ni))

        if ni == word2index['EOS']:
            decoded_words.append('EOS')
            decoder_attentions = decoder_attentions[0:di+1,]
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