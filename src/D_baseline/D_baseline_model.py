
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# the model
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

######################################################################
# The Encoder
# -----------
class EncoderRNN(nn.Module):
	# output is the same dimension as input (dimension defined by externalword embedding model)
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        # self.embeddings_index = embeddings_index

        # self.embedding = nn.Embedding(input_size, input_dim)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden, embeddings_index):
        # input is a word token
        try:
            embedded = Variable(embeddings_index[input].view(1, 1, -1))
        except KeyError:
            embedded = Variable(embeddings_index['UNK'].view(1, 1, -1))
        # embedded = input.view(1,1,-1)
        if use_cuda:
            embedded = embedded.cuda()
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


######################################################################
# multi-layer perceptron
# ^^^^^^^^^^^^^^^^^^^^^^
# code adapted from pytorch tutorial
class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        # maximum input length it can take (for attention mechanism)
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.max_input_len = max_input_len
        
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()

        self.attn = nn.Tanh( nn.Linear(self.input_size+self.hidden_size, 1) )
        self.attn_combine = nn.Linear(self.input_size+self.hidden_size, self.input_size)

    def forward(self, inputs):
        # inputs is a matrix of size (number of tokens in input senquence) * (embedding_dimension)
        attn_weights = F.softmax( attn(inputs) ) # dim = (num of tokens) * 1
        attn_applied = torch.bmm( attn_weights.t().unsqueeze, inputs.unsqueeze(0) ) # new context vector
        out = self.layer1(attn_applied)
        out = self.relu(out)
        out = self.layer2(attn_applied)
        return out

