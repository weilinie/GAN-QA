#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
# the baseline model
#-----------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
from torch.autograd import Variable
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
        embedded = Variable(embeddings_index[input].view(1, 1, -1))

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
    def __init__(self, attn_model, input_size, hidden_size, output_size,
        n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # self.embeddings_index = embeddings_index

        self.attn = nn.Linear(self.input_size+self.hidden_size, 1)
        self.attn_combine = nn.Linear(self.hidden_size+self.hidden_size, self.input_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, embeddings_index):

        # because the number of input tokens varies, we move the init of attn to here
        # instead of in __init__ function
        # attn = nn.Linear(self.input_size+self.hidden_size, encoder_outputs.size()[0])
        # if use_cuda:
        #     attn = attn.cuda()
        
        embedded = Variable(embeddings_index[input].view(1, 1, -1))
        if use_cuda:
            embedded = embedded.cuda()
        # embedded = self.dropout(embedded)

        attn_weights = Variable(torch.zeros(1, encoder_outputs.size()[0])) # length = 1 x length of input tokens

        # for debugging purpose
        print('size of encoder outputs: ' + str(encoder_outputs.size()))
        print('size of encoder output: ' + str(encoder_outputs[1,].size()))
        print('size of hideden: ' + str(hidden[0].size()))
        print('size of attn_weights: ' + str(attn_weights.size()))


        for i in range(encoder_outputs.size()[0]):
            print(i)
            attn_weights[0,i] = F.softmax(self.attn(torch.cat((encoder_outputs[i,], hidden[0]),1).unsqueeze(0)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)) # attn_weights size = 1 x 1 x len input tokens after unsqueeze
 
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

