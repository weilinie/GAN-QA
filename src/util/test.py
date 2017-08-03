# various test cases

# load model
import sys
workspace_path1 = '/home/jack/Documents/QA_QG/GAN-QA/src/util/'
workspace_path2 = '/home/jack/Documents/QA_QG/GAN-QA/src/G_baseline/'
sys.path.insert(0, workspace_path1)
sys.path.insert(0, workspace_path2)
from data_proc import *
from util import *
from G_baseline_model import *
from G_eval import *
import torch


# test these models with data
f_name = 'dev-v1.1.json'
dataset = 'squad'
path_to_dataset = '/home/jack/Documents/QA_QG/data/'
path_to_data = path_to_dataset + dataset + '/' + f_name
exp_name = 'QG_seq2seq_baseline'
path_to_exp_out = '/home/jack/Documents/QA_QG/exp_results/' + exp_name
GLOVE_DIR = path_to_dataset + 'glove.6B/'

encoder1 = torch.load(path_to_exp_out+'/encoder1')
encoder2 = torch.load(path_to_exp_out+'/encoder2')
decoder  = torch.load(path_to_exp_out+'/decoder')

triplets = readSQuAD(path_to_data)

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    coefs = torch.from_numpy(coefs)
    embeddings_index[word] = coefs
f.close()

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

data_tokens = []
for triple in triplets:
    c = post_proc_tokenizer(spacynlp.tokenizer(triple[0]))
    q = post_proc_tokenizer(spacynlp.tokenizer(triple[1]))
    a = post_proc_tokenizer(spacynlp.tokenizer(triple[2]))
    data_tokens += c + q + a
data_tokens = list(set(data_tokens)) # find unique
data_tokens = ['SOS', 'EOS', 'UNK'] + data_tokens

num_tokens = len(data_tokens)
effective_tokens = list(set(data_tokens).intersection(embeddings_index.keys()))
print(effective_tokens[0:20])
effective_num_tokens = len(effective_tokens)


# build word2index dictionary and index2word dictionary
word2index = {}
index2word = {}
for i in range(effective_num_tokens):
    index2word[i] = effective_tokens[i]
    word2index[effective_tokens[i]] = i

# randomly evaluate
max_length = 100
evaluateRandomly(encoder1, encoder2, decoder, triplets, embeddings_index, word2index, index2word, max_length, n=1)