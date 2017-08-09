
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# the model
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../") + '/util')
# print(os.path.abspath(__file__ + '/../../../../')+'/util')
sys.path.append(os.path.abspath(__file__ + "/../../") + '/G_baseline_batch')

from data_proc import *
from model_zoo import *
from util import *

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

######################################################################
# The Encoder
# -----------
if __name__ == '__main__':

    class D(nn.Module):

        def __init__(self, enc_input_size, enc_hidden_size, enc_n_layers, num_directions, batch_size,
                     mlp_hidden_size, num_attn_weights, mlp_output_size, use_attn):

            self.enc_input_size = enc_input_size
            self.enc_hidden_size = enc_hidden_size
            self.enc_n_layers = enc_n_layers
            self.num_directions = num_directions
            self.batch_size = batch_size
            self.encoder = EncoderRNN(enc_input_size, enc_hidden_size, batch_size, enc_n_layers, num_directions)

            self.mlp_input_size = self.enc_hidden_size * self.num_directions
            self.mlp_hidden_size = mlp_hidden_size
            self.num_attn_weights = num_attn_weights
            self.mlp_output_size = mlp_output_size
            self.use_attn = use_attn

            self.mlp = MLP(mlp_input_size, mlp_hidden_size, mlp_output_size, self.encoder, num_attn_weights, use_attn = True)

        def forward(self, inputs, hidden=None):
            # input is tuple (concat of paragraph + answer + question, label). label = question is real or fake (binary)
            # this first part is of size (seq len, batch size, embeddings dim)
            labels = inputs[1]
            inputs = inputs[0]

            # encoding
            # outputs dim (seq_len, batch size, hidden_size*num_directions)
            encoder_outputs, encoder_hidden = self.encoder(inputs, hidden)

            # MLP
            out = self.MLP(encoder_outputs)

            return out, labels
