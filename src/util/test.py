# various test cases

# load model

from data_proc import *
from util import *
sys.path.append(os.path.abspath(__file__ + "/../../") + '/D_baseline')
sys.path.append(os.path.abspath(__file__ + "/../../") + '/G_baseline')
from model_zoo import *
import torch


######################################################################
######################################################################
# test case of get_random_batch and prepare_batch_var functions in data_proc.py
# (uncomment code below to test)
# # test and time
# # to run this test, you need to have these things ready:
# # 1) triplet processed by tokenize_squad,
# # 2) embeddings_index
# # 3) a mini batch processed by get_random_batch
# batch_size = 500
# start = time.time()
# batch, seq_lens, fake_batch, fake_seq_lens = get_random_batch(triplets, batch_size, with_fake=True)
#
# temp, temp_orig, seq_lens_cqa = prepare_batch_var(batch, seq_lens, fake_batch, fake_seq_lens, batch_size, word2index, embeddings_index, embeddings_size,
#                                                   mode = ['word', 'index'], concat_opt='cqa', with_fake=True)
# end = time.time()
# print('time elapsed: ' + str(end-start))
# # the following check if the batched data matches with the original data
# batch_idx = random.choice(range(batch_size))
# print(batch_idx)
#
# print('context  > ', ' '.join(temp_orig[0][batch_idx]))
# print('question > ', ' '.join(temp_orig[1][batch_idx]))
# print('answer   > ', ' '.join(temp_orig[2][batch_idx]))
#
# idx = batch[0].index(temp_orig[0][batch_idx])
# print('context  > ', ' '.join(batch[0][idx]))
# print('question > ', ' '.join(batch[1][idx]))
# print('answer   > ', ' '.join(batch[2][idx]))

# seq_idx = random.choice(range(min(seq_lens[0])))
# print(seq_idx)
# word1 = embeddings_index[batch[0][seq_lens[0].index(heapq.nlargest(batch_idx, seq_lens[0])[-1])][seq_idx]]
# word2 = temp[0][seq_idx, batch_idx,]
# set(word1) == set(word2.data.cpu())





