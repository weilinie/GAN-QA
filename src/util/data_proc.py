#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# data loading helper functions
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import random

import spacy
from spacy.en import English
spacynlp = English()

import torch

import nltk
import json
import numpy as np
import os


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
# read GLOVE word embeddings
def readGlove(path_to_data):
    embeddings_index = {}
    f = open(path_to_data)
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
    print('dimension of word embeddings: ' + str(embeddings_size))
    SOS_token = -torch.ones(embeddings_size) # start of sentence token, all zerons
    EOS_token = torch.ones(embeddings_size) # end of sentence token, all ones
    UNK_token = torch.ones(embeddings_size) + torch.ones(embeddings_size) # these choices are pretty random
    PAD_token = torch.zeros(embeddings_size)
    # add special tokens to the embeddings
    embeddings_index['SOS'] = SOS_token
    embeddings_index['EOS'] = EOS_token
    embeddings_index['UNK'] = UNK_token
    embeddings_index['PAD'] = PAD_token

    return embeddings_index, embeddings_size


######################################################################
# read data specific for SQUAD dataset

def read_raw_squad(path_to_data):
    # output (context, question, answer, ans_start_idx, ans_end_idx) triplets
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
                        ans_start_idx = answers[a]['answer_start']
                        ans_end_idx = ans_start_idx + len(ans_text)
                        triplets.append((context, question, ans_text, ans_start_idx, ans_end_idx))
    return triplets


# helper function to tokenize the raw squad data
# e.g. the context is read as a string; this function produces a list of word tokens from context string
# and return as the processed tuple (context, question, ans_text, ans_start_idx, ans_end_idx)
# the first three are lists, the last two are LongTensor
def tokenize_squad(raw_squad, embeddings_index):
    tokenized_triplets = []
    for triple in raw_squad:
        tokenized_triplets.append( ( tokenize_sentence(triple[0], embeddings_index),
                                     tokenize_sentence(triple[1], embeddings_index),
                                     tokenize_sentence(triple[2], embeddings_index),
                                     torch.FloatTensor([triple[3]]),
                                     torch.FloatTensor([triple[4]]) ) )
    return tokenized_triplets

# turns a sentence into individual tokens
# this function takes care of word tokens that does not appear in pre trained embeddings
# solution is to turn those word tokens into 'UNK'
def tokenize_sentence(sentence, data_tokens):
    tokenized_sentence = spacynlp.tokenizer(sentence)
    # # an additional preprocessing step to separate words and non-words when they appear together
    proc_tokenized_sentence = post_proc_tokenize_sentence(tokenized_sentence)

    token_num = len(proc_tokenized_sentence)

    var = []

    for t in range(0, token_num):
        # the first if loop only for experimental use to aviod large vocab size
        if proc_tokenized_sentence[t] not in data_tokens:
            var.append('UNK')
        else:
            var.append(proc_tokenized_sentence[t])

    var.append('EOS')
    return var


# helper function for post processing tokenizer
# separate all punctuations into single tokens
# e.g. "(they're)" --> "they", "'", "re"
# outputs a list of strings
def post_proc_tokenize_sentence(tokenized_sentence):
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


# find the max length of context, answer, and question
def max_length(triplets):

    max_len_c = 0
    max_len_q = 0
    max_len_a = 0

    for triple in triplets:
        len_c = len(triple[0])
        len_q = len(triple[1])
        len_a = len(triple[2])
        if len_c > max_len_c:
            max_len_c = len_c
        if len_q > max_len_q:
            max_len_q = len_q
        if len_a > max_len_a:
            max_len_a = len_a

    return max_len_c, max_len_q, max_len_a


######################################################################
# count the number of tokens in both the word embeddings and the corpus
def count_effective_num_tokens(triplets, embeddings_index):
    ## find all unique tokens in the data (should be a subset of the number of embeddings)
    data_tokens = []
    for triple in triplets:
        data_tokens += triple[0] + triple[1] + triple[2]
    data_tokens = list(set(data_tokens)) # find unique
    data_tokens = ['SOS', 'EOS', 'UNK', 'PAD'] + data_tokens

    effective_tokens = list(set(data_tokens).intersection(embeddings_index.keys()))
    effective_num_tokens = len(effective_tokens)

    return effective_tokens, effective_num_tokens


######################################################################
# generate word index and index word look up tables
def generate_look_up_table(effective_tokens, effective_num_tokens):
    word2index = {}
    index2word = {}
    for i in range(effective_num_tokens):
        index2word[i] = effective_tokens[i]
        word2index[effective_tokens[i]] = i
    return word2index, index2word


######################################################################
# prepare minibatch of data
# output is (contexts, questions, answers, answer_start_idxs, answer_end_idxs)
# each is of dimension [batch_size x their respective max length]
def get_random_batch(triplets, batch_size, word2index):

    # init values
    questions = []
    contexts_answers = []
    ans_start_idxs = []
    ans_end_idxs = []

    # inside this forloop, all word tokens are turned into their respective index according to word2index lookup table
    for i in range(batch_size):
        triple = random.choice(triplets)
        questions.append( triple[1] )
        contexts_answers.append(triple[0] + triple[2])
        ans_start_idxs.append( triple[3] )
        ans_end_idxs.append( triple[4] )

    # zip and sort by (ascending order) context lengths
    # NOTE: set reverse to be true to satisfy the requirement for a particular pytorch function for mini batch input
    #       preparation. see http://pytorch.org/docs/master/nn.html#torch.nn.utils.rnn.pack_padded_sequence
    seq_triplets = sorted(zip(contexts_answers, questions, ans_start_idxs, ans_end_idxs), key=lambda p: len(p[0]), reverse=True)
    contexts_answers, questions, ans_start_idxs, ans_end_idxs = zip(*seq_triplets)

    # get lengths of each context, question, answer in their respective arrays
    question_lens = [len(s) for s in questions]
    context_answer_lens = [len(s) for s in contexts_answers]

    # pad each context, question, answer to their respective max length
    questions = [pad_sequence(s, max(question_lens), word2index, mode='index') for s in questions]
    contexts_answers = [pad_sequence(s, max(context_answer_lens), word2index) for s in contexts_answers]

    return (contexts_answers, questions, ans_start_idxs, ans_end_idxs), (context_answer_lens, question_lens)


# helper function to zero pad context, question, answer to their respective maximum length
def pad_sequence(s, max_len, word2index, mode = 'word'):
    if mode == 'word':
        return s + ['PAD' for i in range(max_len - len(s))]
    elif mode == 'index':
        return [word2index[i] for i in s] + [word2index['PAD'] for i in range(max_len - len(s))]

# - prepare batch training data
# - training_batch contains five pieces of data. The first three with size [batch size x max seq len],
# - the last two with size [batch size].
# - seq_lens contains lengths of the first three sequences, each of size [batch size]
# - the output would be matrices of size [max seq len x batch size x embedding size]
# - if question is represented as index, then its size is [max seq len x batch size] --> this is transpose of the input
#   from get_random_batch in order to fit NLLLoss function (indexing and selecting the whole batch of a single token) is
#   easier. e.g. you can do question[i] which selects the whole sequence of the first dimension
def prepare_batch_var(batch, seq_lens, batch_size, embeddings_index, embeddings_size, use_cuda=1, question_mode = 'index'):
    context_answer_batch = batch[0]  # size = [max seq len x batch size].
    question_batch = batch[1]

    # init variable matrices
    context_answer_var = torch.zeros(max(seq_lens[0]), batch_size, embeddings_size)
    if question_mode != 'index':
        question_var = torch.zeros(max(seq_lens[1]), batch_size, embeddings_size)
    else:
        question_var = torch.zeros(max(seq_lens[1]), batch_size)

    # TODO very stupid embedded for loop implementation
    for i in range(batch_size):
        for j in range(max(seq_lens[0])):
            context_answer_var[j, i, ] = embeddings_index[context_answer_batch[i][j]]
    if use_cuda:
        context_answer_var = Variable(context_answer_var.cuda())

    for i in range(batch_size):
        for j in range(max(seq_lens[1])):
            if question_mode != 'index':
                question_var[j, i,] = embeddings_index[question_batch[i][j]]
            else:
                question_var[j, i] = question_batch[i][j]
    if use_cuda:
         question_var = Variable(question_var.cuda())

    return (context_answer_var, question_var)

# # test and time
# # to run this test, you need to have these things ready:
# # 1) triplet processed by tokenize_squad,
# # 2) embeddings_index
# # 3) a mini batch processed by get_random_batch
# batch_size = 500
# start = time.time()
# batch, seq_lens = get_random_batch(triplets, batch_size)
# temp = prepare_batch_var(batch, seq_lens, batch_size, embeddings_index, embeddings_size)
# end = time.time()
# print('time elapsed: ' + str(end-start))
# # the following check if the batched data matches with the original data
# batch_idx = random.choice(range(batch_size))
# print(batch_idx)
# seq_idx = random.choice(range(max(seq_lens[0])))
# print(seq_idx)
# word1 = embeddings_index[batch[0][batch_idx][seq_idx]]
# word2 = temp[0][seq_idx, batch_idx,]
# set(word1) == set(word2)



######################################################################
# test function for examining the output of the batch
# primarily see whether the context, question, answer triplets make sense
def print_batch(batch, batch_size, index2word):
    idx = random.choice(range(batch_size))
    context = [ index2word[i] for i in batch[0][idx,] ]
    question = [ index2word[i] for i in batch[1][idx,] ]
    answer = [ index2word[i] for i in batch[2][idx,] ]
    return (' '.join(context), ' '.join(question), ' '.join(answer))



