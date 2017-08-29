#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# data loading helper functions
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import random

# import spacy
from spacy.en import English
spacynlp = English()

import torch
from torch.autograd import Variable

# FIXME: import spacy again below to avoid an error encountered when importing torch and spacy
#        it seems that spacy needs to be imported before torch. However, on Baylor cluster,
#        you need to import spacy again here for it to actually be imported without error.
from spacy.en import English
spacynlp = English()

import json
import numpy as np

# import sys, os
# sys.path.append(os.path.abspath(__file__ + "/../../") + '/G_baseline')
# from G_eval import *


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
                                     triple[3],
                                     triple[4] ) )
    return tokenized_triplets


# helper function to get the sentence of where the answer appear in the context
# based on tokenized_squad, first element in output
# output seq of tokens only from the answer sentence (same format as element in tokenize_squad output)
def get_ans_sentence(tokenized_triplets):
    for tokenized_triple in tokenized_triplets:
        c = tokenized_triple[0]
        ans_start_idx = tokenized_triple[3]
        ans_end_idx = tokenized_triple[4]



# helper function to get a window of tokens around the answer
# similar to get_ans_sentence; only difference is the span of tokens
def get_windowed_ans(raw_squad, window_size):

    windowed_c_triplets = []

    for triple in raw_squad:
        c = triple[0]
        a = triple[2]
        tokenized_c = spacynlp.tokenizer(c)
        # sanity check
        # print(tokenized_c)
        tokenized_a = spacynlp.tokenizer(a)
        ans_start_idx = triple[3]
        ans_end_idx = triple[4]
        c_sub = c[:ans_start_idx]
        print('number of spaces = %d' % c_sub.count(' '))
        
        # find the start token of the answer in context
        idx = 0
        t = 0
        for token in tokenized_c:
            print('current idx = %d' % idx)
            if token == tokenized_a[0]:
                print('found answer token')
                print('current index = %d' % (idx+c_sub.count(' ')))
                print('answer index  = %d' % ans_start_idx)
            if idx+c_sub.count(' ') == ans_start_idx and token == tokenized_a[0]:
                print('entered if statement')
                break
            else:
                idx += len(token)
                t += 1
        print('answe token is %d' % t)
        if t < window_size:
            left_window = 0;
        else:
            left_window = t - window_size
        if t + window_size + len(tokenized_a) > len(tokenized_c):
            right_window = len(tokenized_c)
        else:
            right_window = t + window_size + len(tokenized_a)

        windowed_c = tokenized_c[left_window:right_window]
        # sanity check
        if tokenized_a[0] not in windowed_c:
            print('ERROR: windowed context does not contain answer token')

        windowed_c_triplets.append( ( windowed_c , triple[1], tokenized_a, triple[3], triple[4] ) )

    return windowed_c_triplets




# turns a sentence into individual tokens
# this function takes care of word tokens that does not appear in pre trained embeddings
# solution is to turn those word tokens into 'UNK'
def tokenize_sentence(sentence, data_tokens, spacy=True):
    if spacy:
        tokenized_sentence = spacynlp.tokenizer(sentence)
    else:
        tokenized_sentence = sentence
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


# # find the max length of context, answer, and question
# def max_length(triplets):

#     max_len_c = 0
#     max_len_q = 0
#     max_len_a = 0

#     for triple in triplets:
#         len_c = len(triple[0])
#         len_q = len(triple[1])
#         len_a = len(triple[2])
#         if len_c > max_len_c:
#             max_len_c = len_c
#         if len_q > max_len_q:
#             max_len_q = len_q
#         if len_a > max_len_a:
#             max_len_a = len_a

#     return max_len_c, max_len_q, max_len_a


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
def generate_look_up_table(effective_tokens, effective_num_tokens, use_cuda = True):
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
def get_random_batch(triplets, batch_size, with_fake = False):

    # init values
    contexts = []
    questions = []
    answers = []
    ans_start_idxs = []
    ans_end_idxs = []

    # inside this forloop, all word tokens are turned into their respective index according to word2index lookup table
    for i in range(batch_size):
        triple = random.choice(triplets)
        contexts.append(triple[0])
        questions.append( triple[1] )
        answers.append(triple[2])
        ans_start_idxs.append( triple[3] )
        ans_end_idxs.append( triple[4] )

    # get lengths of each context, question, answer in their respective arrays
    context_lens = [len(s) for s in contexts]
    question_lens = [len(s) for s in questions]
    answer_lens = [len(s) for s in answers]

    if with_fake:
        idx = int(batch_size/2)
        return [contexts[:idx], questions[:idx], answers[:idx], ans_start_idxs[:idx], ans_end_idxs[:idx]], \
               [context_lens[:idx], question_lens[:idx], answer_lens[:idx]],\
               [contexts[idx:], questions, answers[idx:], ans_start_idxs[idx:], ans_end_idxs[idx:]], \
               [context_lens[idx:], question_lens[idx:], answer_lens[idx:]]
    else:
        return [contexts, questions, answers, ans_start_idxs, ans_end_idxs], \
               [context_lens, question_lens, answer_lens]


# - prepare batch training data
# - training_batch contains five pieces of data. The first three with size [batch size x max seq len],
# - the last two with size [batch size].
# - seq_lens contains lengths of the first three sequences, each of size [batch size]
# - the output would be matrices of size [max seq len x batch size x embedding size]
# - if question is represented as index, then its size is [max seq len x batch size] --> this is transpose of the input
#   from get_random_batch in order to fit NLLLoss function (indexing and selecting the whole batch of a single token) is
#   easier. e.g. you can do question[i] which selects the whole sequence of the first dimension
def prepare_batch_var(batch, seq_lens, batch_size, word2index, embeddings_index, embeddings_size,
                      use_cuda=1, mode = ('word', 'word', 'index'), concat_opt = None,
                      with_fake = False, fake_batch = None, fake_seq_lens = None):

    batch_vars = []
    batch_var_orig = []

    if with_fake:
        batch_size = int(batch_size/2)
        fake_q = fake_batch[1]
        fake_q_lens = fake_seq_lens[1]

    #TODO (for different applications): change the below code (before for loop) to concat different portions of the batch_triplets
    if concat_opt == None:
        pass

    elif concat_opt == 'ca':
        ca = []
        ca_len = []
        for b in range(batch_size):
            ca.append(batch[0][b] + batch[2][b])
            ca_len.append(len(batch[0][b] + batch[2][b]))
        batch = [ca, batch[1], batch[3], batch[4]]
        seq_lens =  [ca_len] + seq_lens

    elif concat_opt == 'qa':
        pass

    # FIXME: only this following elif implemented fake question
    elif concat_opt == 'cqa':
        cqa = []
        cqa_len = []
        labels = []
        for b in range(batch_size):
            cqa.append(batch[0][b] + batch[1][b] + batch[2][b]) # append real
            cqa_len.append(len(batch[0][b] + batch[1][b] + batch[2][b])) # append real
            labels.append(1)
            if with_fake: # append fake
                fake_q_sample = random.sample(fake_q,1)[0]
                cqa.append(batch[0][b] + fake_q_sample + batch[2][b])
                cqa_len.append(len(batch[0][b] + fake_q_sample + batch[2][b]))
                labels.append(0)
        if with_fake:
            batch = [cqa, batch[3]+fake_batch[3], batch[4]+fake_batch[4], labels]
        else:
            batch = [cqa, batch[3], batch[4]]
        seq_lens = [cqa_len]
    elif concat_opt == 'qca':
        pass

    else:
        raise ValueError('not a valid concat option.')

    # sort this batch_var in descending order according to the values of the lengths of the first element in batch
    num_batch = len(batch)
    all = batch + seq_lens
    all = sorted(zip(*all), key=lambda p: len(p[0]), reverse=True)
    all = zip(*all)
    batch = all[0:num_batch]
    seq_lens = all[num_batch:]
    batch_orig = batch

    # get bacth size back to 2x if with fake
    if with_fake:
        batch_size = batch_size * 2

    for b in range(num_batch):

        batch_var = batch[b]

        # if element in batch is float, i.e. indices, then do nothing
        if isinstance(batch_var[0], int):
            batch_var = list(batch_var)
            pass
        else:
            # pad each context, question, answer to their respective max length
            if mode[b]  == 'index':
                batch_padded = [pad_sequence(s, max(seq_lens[b]), word2index, mode='index') for s in batch_var]
            else:
                batch_padded = [pad_sequence(s, max(seq_lens[b]), word2index) for s in batch_var]

            # init variable matrices
            if mode[b] == 'index':
                batch_var = torch.LongTensor(max(seq_lens[b]), batch_size) # long tensor for module loss criterion
            else:
                batch_var = torch.FloatTensor(max(seq_lens[b]), batch_size, embeddings_size)

            # FIXME: very stupid embedded for loop implementation
            for i in range(batch_size):
                for j in range(max(seq_lens[b])):
                    if mode[b] == 'index':
                        batch_var[j, i] = batch_padded[i][j]
                    else:
                        batch_var[j, i,] = embeddings_index[batch_padded[i][j]]

        batch_vars.append(batch_var)

    # the second output is for debugging purpose
    return batch_vars, batch_orig, seq_lens

# helper function to zero pad context, question, answer to their respective maximum length
def pad_sequence(s, max_len, word2index, mode = 'word'):
    if mode == 'word':
        return s + ['PAD' for i in range(max_len - len(s))]
    elif mode == 'index':
        return [word2index[i] for i in s] + [word2index['PAD'] for i in range(max_len - len(s))]


######################################################################
# TODO: need a function to sample some (c, q, a) triplets from the generator
def sample_generated_triples(triplets, G, batch_size):

    # should return the same thing as get_random_batch with with_fake = False
    return None


######################################################################
# test function for examining the output of the batch
# primarily see whether the context, question, answer triplets make sense
def print_batch(batch, batch_size, index2word):
    idx = random.choice(range(batch_size))
    context = [ index2word[i] for i in batch[0][idx,] ]
    question = [ index2word[i] for i in batch[1][idx,] ]
    answer = [ index2word[i] for i in batch[2][idx,] ]
    return (' '.join(context), ' '.join(question), ' '.join(answer))



