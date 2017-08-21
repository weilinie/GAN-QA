
import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../"))
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')

# FIXME: spacy has some problem with torch. need to import spacy first. therefore import data_proc first.
from data_proc import *
from model_zoo import *

import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()

######################################################################
# The Encoder
# -----------
# FIXME: not sure if __name__ is to be used. 
# if __name__ == '__main__':

class D(nn.Module):

    def __init__(self, enc_input_size, enc_hidden_size, enc_n_layers, num_directions,
                 mlp_hidden_size, num_attn_weights, mlp_output_size, use_attn,
                 batch_size):
        # super constructor
        super(D, self).__init__()

        self.encoder = EncoderRNN(enc_input_size, enc_hidden_size, batch_size, enc_n_layers, num_directions)
        self.mlp = MLP(mlp_hidden_size, mlp_output_size, self.encoder, num_attn_weights, use_attn = True)


    def forward(self, inputs, seq_lens, hidden=None):
        # input size = (seq len, batch size, word embedding dimension)
        
        # encoding
        # outputs dim (seq_len, batch size, hidden_size*num_directions)
        encoder_outputs, encoder_hidden = self.encoder(inputs, seq_lens)

        # MLP
        out = self.mlp(encoder_outputs)

        return out


    def backward(self, out, labels, criterion, optimizer):
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        return loss



