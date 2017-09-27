#-----------------------------------------------------------------------------------------------#
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import random

import torch
from torch.autograd import Variable

import json
import numpy as np
import random

# import sys, os
# sys.path.append(os.path.abspath(__file__ + "/../../") + '/G_baseline')
# from G_eval import *


######################################################################
# read data specific for SQUAD dataset
def read_raw_squad(path_to_data, normalize=True):
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
                qas = samples[p]['qas']
                for i in range(0, len(qas)):
                # print('current s,p,i are: ' + str(s)+str(p)+str(i))
                    answers = qas[i]['answers']
                    question = qas[i]['question']
                    for a in range(0, len(answers)):
                        ans_text = answers[a]['text']
                        ans_start_idx = answers[a]['answer_start']
                        ans_end_idx = ans_start_idx + len(ans_text)
                        triplets.append((context, question, ans_text, ans_start_idx, ans_end_idx))
    return triplets


######################################################################
# check whether each example contains a non english character
# https://stackoverflow.com/questions/27084617/detect-strings-with-non-english-characters-in-python
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

# function to read lines from file
def readLinesFromFile(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.rstrip('\n') for x in content]
    f.close()
    return content

######################################################################
## scripts

dataset = 'squad'
f_name = 'train-v1.1.json'
path_to_dataset = '/home/jack/Documents/QA_QG/data/'
path_to_data = path_to_dataset + dataset + '/' + f_name
with open(path_to_data) as f:
    raw_data = json.load(f)
raw_triplets = read_raw_squad(path_to_data)

# ## examine corpus character frequencies
# char_freq_dict = {}
# c_char_freq_dict = {}
# q_char_freq_dict = {}
# a_char_freq_dict = {}
# triple_is_English = []
# for triple in raw_triplets:
#     c = triple[0]
#     q = triple[1]
#     a = triple[2]
#     for i in c:
#         if i in c_char_freq_dict.keys():
#             c_char_freq_dict[i] += 1
#         else:
#             c_char_freq_dict[i] = 1
#         if i in char_freq_dict.keys():
#             char_freq_dict[i] += 1
#         else:
#             char_freq_dict[i] = 1
#     for i in q:
#         if i in q_char_freq_dict.keys():
#             q_char_freq_dict[i] += 1
#         else:
#             q_char_freq_dict[i] = 1
#         if i in char_freq_dict.keys():
#             char_freq_dict[i] += 1
#         else:
#             char_freq_dict[i] = 1
#     for i in a:
#         if i in a_char_freq_dict.keys():
#             a_char_freq_dict[i] += 1
#         else:
#             a_char_freq_dict[i] = 1
#         if i in char_freq_dict.keys():
#             char_freq_dict[i] += 1
#         else:
#             char_freq_dict[i] = 1
#     triple_is_English.append(isEnglish(c))

## check if an example (by looking at context) contains non english characters
is_english = []
for triple in raw_triplets:
    c = triple[0]
    is_english.append(isEnglish(c))
f = open('/home/jack/Documents/QA_QG/data/processed_squad/is_english.txt', 'w')
for a in is_english:
    # f.write(unicode(a)+'\n') # for python 2
    f.write(a+'\n') # for python 3

## exclude all those triples that are 1) not english 2) contain escape characters
new_triplets = {}
new_triplets['questions'] = []
new_triplets['contexts'] = []
new_triplets['answers'] = []
new_triplets['ans_start_idx'] = []
new_triplets['ans_end_idx'] = []
for i in range(len(raw_triplets)):
    c = raw_triplets[i][0]
    q = raw_triplets[i][1]
    a = raw_triplets[i][2]
    if is_english[i] and '\n' not in c and '\n' not in q and '\n' not in a:
        # new_triplets.append(raw_triplets[i])
        new_triplets['contexts'].append(c)
        new_triplets['questions'].append(q)
        new_triplets['answers'].append(a)
        new_triplets['ans_start_idx'].append(raw_triplets[i][3])
        new_triplets['ans_end_idx'].append(raw_triplets[i][4])
with open('/home/jack/Documents/QA_QG/data/processed_squad/squad_EnglishOnly_noEscape.json', 'w') as outfile:
    json.dump(new_triplets, outfile)
# test load
with open('/home/jack/Documents/QA_QG/data/processed_squad/squad_EnglishOnly_noEscape.json') as f:
    new_triplets_read = json.load(f)
# write these triplets to file (separate files, c, q, a, a_start_idx, a_end_idx)


##################################################################################
# test to read processed squad by lua script
data_openNMT_path = '/home/jack/Documents/QA_QG/data/squad_openNMT/'
processed_squad_path = '/home/jack/Documents/QA_QG/data/processed_squad/'

# get english only, no escape char squad (in separate files)
new_contexts = readLinesFromFile(processed_squad_path+'contexts_EnglishOnly_noEscape.txt')
new_questions = readLinesFromFile(processed_squad_path+'questions_EnglishOnly_noEscape.txt')
new_answers = readLinesFromFile(processed_squad_path+'answers_EnglishOnly_noEscape.txt')
new_a_start_idxs = readLinesFromFile(processed_squad_path+'a_start_idxs_EnglishOnly_noEscape.txt')
new_ans_end_idxs = readLinesFromFile(processed_squad_path+'a_end_idxs_EnglishOnly_noEscape.txt')
# print lengths
print(len(new_contexts))
print(len(new_questions))
print(len(new_answers))
print(len(new_a_start_idxs))
print(len(new_a_end_idxs))

# get_ans_token_idx
tokenized_contexts = readLinesFromFile(data_openNMT_path+'cs_min_NoAnnotate.txt')
tokenized_questions = readLinesFromFile(data_openNMT_path+'qs_min_NoAnnotate.txt')
tokenized_answers = readLinesFromFile(data_openNMT_path+'as_min_NoAnnotate.txt')
# a_start_idxs = readLinesFromFile(data_openNMT_path+'ans_start_idx.txt')
# a_end_idxs = readLinesFromFile(data_openNMT_path+'ans_end_idx.txt')
a_token_start_idxs = readLinesFromFile(data_openNMT_path+'atsi_min.txt')
a_token_end_idxs = readLinesFromFile(data_openNMT_path+'atei_min.txt')

# sanity check: token answer match
# mistmatch should be empty
mismatch = []
for i in range(len(tokenized_contexts)):
    s = int(a_token_start_idxs[i])
    c_w = tokenized_contexts[i].split(' ')
    a_w = tokenized_answers[i].split(' ')
    if c_w[s-1] != a_w[0]:
        mismatch.append(i)