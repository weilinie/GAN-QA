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
# extract the sentence containing the answer
def get_ans_sentence(c, a, asi, aei, sent_window=0):
    sent_cs = []  # now each context in
    is_ans_token_vec = []
    unmatch = []  # for debug
    for t in range(len(c)):
        sent = None
        split = False
        sent_c = list(spacynlp(c[t]).sents)
        tokenized_a = spacynlp.tokenizer(a[t])
        # sanity check
        # if len(sent_c) == 1:
        #     print('WARNING: sentence segmentation may not work in this triple')
        #     print(sent_c)
        # print(tokenized_c)
        ans_start_idx = int(asi[t])
        ans_end_idx = int(aei[t])

        # print(ans_start_idx)
        # print(ans_end_idx)
        idx = 0
        for i in range(len(sent_c)):
            s = sent_c[i]
            if idx <= ans_start_idx and idx + len(s.string) >= ans_start_idx:
                ans_sent_idx = i
                sent = s.string
                break
            else:
                idx += len(s.string)

        if sent is None:
            # print(tokenized_a[0])
            # print(sent)
            unmatch.append(t)

        # TODO: multiple sentences as context (only include the sentence preceeding the answer sentence)
        if sent_window > 0:
            for i in range(1, sent_window):
                if not split:
                    if ans_sent_idx - i > 0 and ans_sent_idx + i < len(sent_c):
                        sent = sent_c[ans_sent_idx - i].string + sent + sent_c[ans_sent_idx + i].string
                    elif ans_sent_idx - i <= 0 and ans_sent_idx + i < len(sent_c):
                        sent = sent + sent_c[ans_sent_idx + i].string
                    elif ans_sent_idx - i > 0 and ans_sent_idx + i >= len(sent_c):
                        sent = sent_c[ans_sent_idx - i].string + sent
                else:
                    if ans_sent_idx - i > 0 and ans_sent_idx + i + 1 < len(sent_c):
                        sent = sent_c[ans_sent_idx - i].string + sent + sent_c[ans_sent_idx + i + 1].string
                    elif ans_sent_idx - i <= 0 and ans_sent_idx + i + 1 < len(sent_c):
                        sent = sent + sent_c[ans_sent_idx + i].string
                    elif ans_sent_idx - i > 0 and ans_sent_idx + i + 1 >= len(sent_c):
                        sent = sent_c[ans_sent_idx - i].string + sent

        tokenized_c = spacynlp.tokenizer(sent)
        is_ans_token = [0] * len(tokenized_c)
        for i in range(len(tokenized_c)):
            token = tokenized_c[i]
            if len(tokenized_c[0:i].string) == ans_start_idx:
                is_ans_token[i:i + len(tokenized_a)] = [1] * len(tokenized_a)
                break

        sent_cs.append(sent)
        is_ans_token_vec.append(is_ans_token)

    # return list(set(unmatch))
    return sent_cs, is_ans_token_vec, list(set(unmatch))

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
    with open(path, encoding='utf-8') as f:
        content = f.readlines()
    content = [x.rstrip('\n') for x in content]
    f.close()
    return content

# concat answers and contexts
def concat_c_a(context, answer):
    pass

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
# f = open('/home/jack/Documents/QA_QG/data/processed_squad/is_english.txt', 'w')
# for a in is_english:
#     # f.write(unicode(a)+'\n') # for python 2
#     f.write(a+'\n') # for python 3

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
with open('/home/jack/Documents/QA_QG/data/processed_squad/dev_squad_EnglishOnly_noEscape.json', 'w') as outfile:
    json.dump(new_triplets, outfile)
# test load
with open('/home/jack/Documents/QA_QG/data/processed_squad/dev_squad_EnglishOnly_noEscape.json') as f:
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
new_a_end_idxs = readLinesFromFile(processed_squad_path+'a_end_idxs_EnglishOnly_noEscape.txt')
# print lengths
print(len(new_contexts))
print(len(new_questions))
print(len(new_answers))
print(len(new_a_start_idxs))
print(len(new_a_end_idxs))

# get_ans_token_idx
tokenized_contexts = readLinesFromFile(data_openNMT_path+'preparation/train/tokenized_NoAnnotate/contexts_EnglishOnly_noEscape_NoAnnotate.txt')
tokenized_questions = readLinesFromFile(data_openNMT_path+'preparation/train/tokenized_NoAnnotate/questions_EnglishOnly_noEscape_NoAnnotate.txt')
tokenized_answers = readLinesFromFile(data_openNMT_path+'preparation/train/tokenized_NoAnnotate/answers_EnglishOnly_noEscape_NoAnnotate.txt')
detokenized_contexts = readLinesFromFile(data_openNMT_path+'preparation/train/detokenized/contexts_EnglishOnly_noEscape_detokenized.txt')
detokenized_questions = readLinesFromFile(data_openNMT_path+'preparation/train/detokenized/questions_EnglishOnly_noEscape_detokenized.txt')
detokenized_answers = readLinesFromFile(data_openNMT_path+'preparation/train/detokenized/answers_EnglishOnly_noEscape_detokenized.txt')
a_start_idxs = readLinesFromFile(data_openNMT_path+'preparation/train/a_start_idxs_EnglishOnly_noEscape.txt')
a_end_idxs = readLinesFromFile(data_openNMT_path+'preparation/train/a_end_idxs_EnglishOnly_noEscape.txt')
a_token_start_idxs = readLinesFromFile(data_openNMT_path+'preparation/train/ans_token_start_idxs.txt')
a_token_end_idxs = readLinesFromFile(data_openNMT_path+'preparation/train/ans_token_end_idxs.txt')

# sanity check: token answer match
# mistmatch should be empty
mismatch = []
for i in range(len(tokenized_contexts)):
    s = int(a_token_start_idxs[i])
    e = int(a_token_end_idxs[i])
    c_w = tokenized_contexts[i].split(' ')
    a_w = tokenized_answers[i].split(' ')
    if s == -1:
        mismatch.append(i)
    else:
        for x in range(e-s+1):
            if c_w[x+s-1] != a_w[x]:
                # print(x)
                mismatch.append(i)
mismatch = list(set(mismatch))
# write those that match complelely to file
f = open(data_openNMT_path+'train/cs_min_NoAnnotate.txt', 'w')
for i in range(len(tokenized_contexts)):
    if i not in mismatch:
        f.write(tokenized_contexts[i]+'\n')
f.close()
f = open(data_openNMT_path+'train/cs_min_detokenized.txt', 'w')
for i in range(len(detokenized_contexts)):
    if i not in mismatch:
        f.write(detokenized_contexts[i]+'\n')
f.close()
f = open(data_openNMT_path+'train/qs_min_NoAnnotate.txt', 'w')
for i in range(len(tokenized_questions)):
    if i not in mismatch:
        f.write(tokenized_questions[i]+'\n')
f.close()
f = open(data_openNMT_path+'train/qs_min_detokenized.txt', 'w')
for i in range(len(detokenized_questions)):
    if i not in mismatch:
        f.write(detokenized_questions[i]+'\n')
f.close()
f = open(data_openNMT_path+'train/as_min_NoAnnotate.txt', 'w')
for i in range(len(tokenized_answers)):
    if i not in mismatch:
        f.write(tokenized_answers[i]+'\n')
f.close()
f = open(data_openNMT_path+'train/as_min_detokenized.txt', 'w')
for i in range(len(detokenized_answers)):
    if i not in mismatch:
        f.write(detokenized_answers[i]+'\n')
f.close()
f = open(data_openNMT_path+'train/atsi_min.txt', 'w')
for i in range(len(a_token_start_idxs)):
    if i not in mismatch:
        f.write(a_token_start_idxs[i]+'\n')
f.close()
f = open(data_openNMT_path+'train/atei_min.txt', 'w')
for i in range(len(a_token_end_idxs)):
    if i not in mismatch:
        f.write(a_token_end_idxs[i]+'\n')
f.close()
f = open(data_openNMT_path+'train/asi_min.txt', 'w')
for i in range(len(a_start_idxs)):
    if i not in mismatch:
        f.write(a_start_idxs[i]+'\n')
f.close()
f = open(data_openNMT_path+'train/aei_min.txt', 'w')
for i in range(len(a_end_idxs)):
    if i not in mismatch:
        f.write(a_end_idxs[i]+'\n')
f.close()

# split dev set to dev and test set according to a portion split and write to file
tokenized_contexts = readLinesFromFile(data_openNMT_path+'preparation/dev/tokenized_NoAnnotate/dev_contexts_EnglishOnly_noEscape_NoAnnotate.txt')
tokenized_questions = readLinesFromFile(data_openNMT_path+'preparation/dev/tokenized_NoAnnotate/dev_questions_EnglishOnly_noEscape_NoAnnotate.txt')
tokenized_answers = readLinesFromFile(data_openNMT_path+'preparation/dev/tokenized_NoAnnotate/dev_answers_EnglishOnly_noEscape_NoAnnotate.txt')
detokenized_contexts = readLinesFromFile(data_openNMT_path+'preparation/dev/detokenized/dev_contexts_EnglishOnly_noEscape_detokenized.txt')
detokenized_questions = readLinesFromFile(data_openNMT_path+'preparation/dev/detokenized/dev_questions_EnglishOnly_noEscape_detokenized.txt')
detokenized_answers = readLinesFromFile(data_openNMT_path+'preparation/dev/detokenized/dev_answers_EnglishOnly_noEscape_detokenized.txt')
a_start_idxs = readLinesFromFile(data_openNMT_path+'preparation/dev/dev_a_start_idxs_EnglishOnly_noEscape.txt')
a_end_idxs = readLinesFromFile(data_openNMT_path+'preparation/dev/dev_a_end_idxs_EnglishOnly_noEscape.txt')
a_token_start_idxs = readLinesFromFile(data_openNMT_path+'preparation/dev/dev_ans_token_start_idxs.txt')
a_token_end_idxs = readLinesFromFile(data_openNMT_path+'preparation/dev/dev_ans_token_end_idxs.txt')
import math
dev_portion = 0.7
dev_range = math.ceil(len(tokenized_contexts)*dev_portion)
# write dev
f = open(data_openNMT_path+'dev/dev_cs_min_NoAnnotate.txt', 'w')
for i in range(dev_range):
    if i not in mismatch:
        f.write(tokenized_contexts[i]+'\n')
f.close()
f = open(data_openNMT_path+'dev/dev_cs_min_detokenized.txt', 'w')
for i in range(dev_range):
    if i not in mismatch:
        f.write(detokenized_contexts[i]+'\n')
f.close()
f = open(data_openNMT_path+'dev/dev_qs_min_NoAnnotate.txt', 'w')
for i in range(dev_range):
    if i not in mismatch:
        f.write(tokenized_questions[i]+'\n')
f.close()
f = open(data_openNMT_path+'dev/dev_qs_min_detokenized.txt', 'w')
for i in range(dev_range):
    if i not in mismatch:
        f.write(detokenized_questions[i]+'\n')
f.close()
f = open(data_openNMT_path+'dev/dev_as_min_NoAnnotate.txt', 'w')
for i in range(dev_range):
    if i not in mismatch:
        f.write(tokenized_answers[i]+'\n')
f.close()
f = open(data_openNMT_path+'dev/dev_as_min_detokenized.txt', 'w')
for i in range(dev_range):
    if i not in mismatch:
        f.write(detokenized_answers[i]+'\n')
f.close()
f = open(data_openNMT_path+'dev/dev_atsi_min.txt', 'w')
for i in range(dev_range):
    if i not in mismatch:
        f.write(a_token_start_idxs[i]+'\n')
f.close()
f = open(data_openNMT_path+'dev/dev_atei_min.txt', 'w')
for i in range(dev_range):
    if i not in mismatch:
        f.write(a_token_end_idxs[i]+'\n')
f.close()
f = open(data_openNMT_path+'dev/dev_asi_min.txt', 'w')
for i in range(dev_range):
    if i not in mismatch:
        f.write(a_start_idxs[i]+'\n')
f.close()
f = open(data_openNMT_path+'dev/dev_aei_min.txt', 'w')
for i in range(dev_range):
    if i not in mismatch:
        f.write(a_end_idxs[i]+'\n')
f.close()
# write test
f = open(data_openNMT_path+'test/test_cs_min_NoAnnotate.txt', 'w')
for i in range(dev_range, len(tokenized_contexts)):
    if i not in mismatch:
        f.write(tokenized_contexts[i]+'\n')
f.close()
f = open(data_openNMT_path+'test/test_cs_min_detokenized.txt', 'w')
for i in range(dev_range, len(detokenized_contexts)):
    if i not in mismatch:
        f.write(detokenized_contexts[i]+'\n')
f.close()
f = open(data_openNMT_path+'test/test_qs_min_NoAnnotate.txt', 'w')
for i in range(dev_range, len(tokenized_questions)):
    if i not in mismatch:
        f.write(tokenized_questions[i]+'\n')
f.close()
f = open(data_openNMT_path+'test/test_qs_min_detokenized.txt', 'w')
for i in range(dev_range, len(detokenized_questions)):
    if i not in mismatch:
        f.write(detokenized_questions[i]+'\n')
f.close()
f = open(data_openNMT_path+'test/test_as_min_NoAnnotate.txt', 'w')
for i in range(dev_range, len(tokenized_answers)):
    if i not in mismatch:
        f.write(tokenized_answers[i]+'\n')
f.close()
f = open(data_openNMT_path+'test/test_as_min_detokenized.txt', 'w')
for i in range(dev_range, len(detokenized_answers)):
    if i not in mismatch:
        f.write(detokenized_answers[i]+'\n')
f.close()
f = open(data_openNMT_path+'test/test_atsi_min.txt', 'w')
for i in range(dev_range, len(a_token_start_idxs)):
    if i not in mismatch:
        f.write(a_token_start_idxs[i]+'\n')
f.close()
f = open(data_openNMT_path+'test/test_atei_min.txt', 'w')
for i in range(dev_range, len(a_token_end_idxs)):
    if i not in mismatch:
        f.write(a_token_end_idxs[i]+'\n')
f.close()
f = open(data_openNMT_path+'test/test_asi_min.txt', 'w')
for i in range(dev_range, len(a_start_idxs)):
    if i not in mismatch:
        f.write(a_start_idxs[i]+'\n')
f.close()
f = open(data_openNMT_path+'test/test_aei_min.txt', 'w')
for i in range(dev_range, len(a_end_idxs)):
    if i not in mismatch:
        f.write(a_end_idxs[i]+'\n')
f.close()


## produce a file containing only answer sentences
temp_c = readLinesFromFile(data_openNMT_path+'test/test_cs_min_detokenized.txt')
temp_a = readLinesFromFile(data_openNMT_path+'test/test_as_min_detokenized.txt')
temp_tc = readLinesFromFile(data_openNMT_path+'test/test_cs_min_NoAnnotate.txt')
temp_ta = readLinesFromFile(data_openNMT_path+'test/test_as_min_NoAnnotate.txt')
temp_asi = readLinesFromFile(data_openNMT_path+'test/test_asi_min.txt')
temp_aei = readLinesFromFile(data_openNMT_path+'test/test_aei_min.txt')
temp_atsi = readLinesFromFile(data_openNMT_path+'test/test_atsi_min.txt')
temp_atei = readLinesFromFile(data_openNMT_path+'test/test_atei_min.txt')
temp_mismatch = []
sent_cs, is_ans_token_vec, unmatch = get_ans_sentence(temp_c, temp_a, temp_asi, temp_aei, sent_window)
f = open(data_openNMT_path+'test/test_cs_min_sent.txt', 'w')
for i in range(len(sent_cs)):
    f.write(sent_cs[i]+'\n')
f.close()

## various tests
# test whether answer extracted using context and indices match with the ground truth answer
temp_c = readLinesFromFile(data_openNMT_path+'train/cs_min_detokenized.txt')
temp_a = readLinesFromFile(data_openNMT_path+'train/as_min_detokenized.txt')
temp_tc = readLinesFromFile(data_openNMT_path+'train/cs_min_NoAnnotate.txt')
temp_ta = readLinesFromFile(data_openNMT_path+'train/as_min_NoAnnotate.txt')
temp_asi = readLinesFromFile(data_openNMT_path+'train/asi_min.txt')
temp_aei = readLinesFromFile(data_openNMT_path+'train/aei_min.txt')
temp_atsi = readLinesFromFile(data_openNMT_path+'train/atsi_min.txt')
temp_atei = readLinesFromFile(data_openNMT_path+'train/atei_min.txt')
temp_mismatch = []
# test
for i in range(len(temp_c)):
    if temp_c[i][int(temp_asi[i]):int(temp_aei[i])] != temp_a[i]:
        temp_mismatch.append(i)
# take a look at the original context sentence and see what is the difference
# usually error by a space (detokenzied sentence is more correct!!)
for i in range(len(new_triplets['contexts'])):
    if 'as an unmarried young woman Victoria was' in new_triplets['contexts'][i]:
        print(i)
# manually correct those indices
# NOTE: those are file specific, do not run the following without looking at temp_mismatch array
temp_aei[temp_mismatch[0]] = str(int(temp_aei[temp_mismatch[0]]) - 1)
temp_aei[temp_mismatch[1]] = str(int(temp_aei[temp_mismatch[1]]) - 1)
f = open(data_openNMT_path+'train/asi_min.txt', 'w')
for i in range(len(temp_asi)):
    f.write(temp_asi[i]+'\n')
f.close()
f = open(data_openNMT_path+'train/aei_min.txt', 'w')
for i in range(len(temp_aei)):
    f.write(temp_aei[i]+'\n')
f.close()
