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
# read data specific for SQUAD dataset

def readSQuAD(path_to_data):
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
                        ans_start_idx = answer[a]['answer_start']
                        ans_end_idx = answer[a][ans_start_idx + len(ans_text)]
                        triplets.append((context, question, ans_text, ans_start_idx, ans_end_idx))
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
    ans_start_idx = torch.LongTensor([triple[3]])
    ans_end_idx  = torch.LongTensor([triple[4]])
    return (context, question, answer, ans_start_idx, ans_end_idx)