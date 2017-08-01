#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# the baseline model
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


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
        embedded = Variable(embeddings_index[input].view(1, 1, -1))
        # try:
        #     embedded = Variable(embeddings_index[input].view(1, 1, -1))
        # except KeyError:
        #     embedded = Variable(embeddings_index['UNK'].view(1, 1, -1))
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
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
        n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # self.embeddings_index = embeddings_index

        # self.attn = nn.Linear(self.input_size+self.hidden_size, self.enc_output_len)
        self.attn_combine = nn.Linear(self.input_size+self.hidden_size, self.input_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs, embeddings_index):

        # because the number of input tokens varies, we move the init of attn to here
        # instead of in __init__ function
        attn = nn.Linear(self.input_size+self.hidden_size, encoder_outputs.size()[0])
        if use_cuda:
            attn = attn.cuda()
        
        embedded = Variable(embeddings_index[input].view(1, 1, -1))
        # try:
        #     embedded = Variable(embeddings_index[input].view(1, 1, -1))
        # except KeyError:
        #     embedded = Variable(embeddings_index['UNK'].view(1, 1, -1))
        # embedded = input.view(1,1,-1)
        if use_cuda:
            embedded = embedded.cuda()
        # embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result