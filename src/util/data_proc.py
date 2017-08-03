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
    # add special tokens to the embeddings
    embeddings_index['SOS'] = SOS_token
    embeddings_index['EOS'] = EOS_token
    embeddings_index['UNK'] = UNK_token

    return embeddings_index, embeddings_size


######################################################################
# read data specific for SQUAD dataset

def readSQuAD(path_to_data, embeddings_index):
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
                context = tokenize_sentence(context, embeddings_index)
                qas = samples[p]['qas']
                for i in range(0, len(qas)):
                # print('current s,p,i are: ' + str(s)+str(p)+str(i))
                    answers = qas[i]['answers']
                    question = qas[i]['question']
                    # turn from unicode to ascii and lower case everything
                    question = normalizeString(question)
                    question = tokenize_sentence(question, embeddings_index)
                    for a in range(0, len(answers)):
                        ans_text = answers[a]['text']
                        # turn from unicode to ascii and lower case everything
                        ans_text = normalizeString(ans_text)
                        ans_text = tokenize_sentence(ans_text, embeddings_index)
                        ans_start_idx = torch.LongTensor(answers[a]['answer_start'])
                        ans_end_idx = torch.LongTensor(ans_start_idx + len(ans_text))
                        triplets.append((context, question, ans_text, ans_start_idx, ans_end_idx))
    return triplets


# turns a sentence into individual tokens
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
        c = post_proc_tokenize_sentence(spacynlp.tokenizer(triple[0]))
        q = post_proc_tokenize_sentence(spacynlp.tokenizer(triple[1]))
        a = post_proc_tokenize_sentence(spacynlp.tokenizer(triple[2]))
        data_tokens += c + q + a
    data_tokens = list(set(data_tokens)) # find unique
    data_tokens = ['SOS', 'EOS', 'UNK'] + data_tokens

    num_tokens = len(data_tokens)
    effective_tokens = list(set(data_tokens).intersection(embeddings_index.keys()))
    print(effective_tokens[0:20])
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





