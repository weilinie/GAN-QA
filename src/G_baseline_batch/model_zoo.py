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
    def __init__(self, input_size, hidden_size, n_layers=1, num_directions=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_directions = num_directions
        # self.embeddings_index = embeddings_index

        # self.embedding = nn.Embedding(input_size, input_dim)
        if self.num_directions == 1:
            self.gru = nn.GRU(input_size, hidden_size, n_layers, bidirectional=False)
        elif self.num_directions == 2:
            self.gru = nn.GRU(input_size, hidden_size, n_layers, bidirectional=True)
        else:
            raise Exception('input num_directions is wrong - need to be either 1 or 2')

    def forward(self, input, hidden=None):
        # input is matrix of size [max seq len x batch size x embedding dimension]

        encoder_outputs, hidden = self.gru(input, hidden)

        # unpack the sequence
        encoder_outputs, output_lens = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs)

        # TODO: do I need to sum the eocnder_outputs when the network is bidirectional:
        # e.g. outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        return encoder_outputs, hidden


######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, encoder, n_layers=1, num_directions = 1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.num_directions = num_directions
        # self.embeddings_index = embeddings_index

        if encoder.bidirec:
            self.attn = nn.Linear(2 * encoder.hidden_size, self.hidden_size)
            self.attn_combine = nn.Linear(self.input_size + encoder.num_directions * encoder.hidden_size, self.input_size)
        else:
            self.attn = nn.Linear(self.hidden_size + encoder.hidden_size, self.hidden_size)
            self.attn_combine = nn.Linear(self.input_size + encoder.hidden_size, self.input_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    # forward for each time step.
    # need to do this because of teacher forcing at each time step
    def forward(self, input, encoder_outputs, embeddings_index, hidden=None):

        # get the output
        # hidden: (num_layers * num_directions, batch, hidden_size)
        decoder_output, hidden = self.gru(input, hidden)

        # # unpack the sequence
        # # decoder_outputs size (seq len, batch, hidden_size * num_directions)
        # # --> collection of hidden states at every time step
        # decoder_outputs, output_lens = torch.nn.utils.rnn.pad_packed_sequence(decoder_outputs)

        # init attention weights
        # length = batch_size x encoder output lens
        attn_weights = Variable(torch.zeros(encoder_outputs.size(1), encoder_outputs.size(0)))
        if use_cuda:
            attn_weights = attn_weights.cuda()

        # calculate attention weight for each output time step
        # remember encoder_outputs size: (seq_len, batch, hidden_size * num_directions)
        # for each token in the decoder output sequences:

        for b in range(encoder_outputs.size(1)):
            # copy the decoder output at the present time step to N rows, where N = num encoder outputs
            # first dimension of append = first dimension of encoder_outputs[:,b] = seq_len of encoder
            append = decoder_output[i, b].repeat(encoder_outputs.size(0),1)
            # the scores for calculating attention weights of all encoder outputs for one time step of decoder output
            scores = torch.mm( decoder_output[i, b].unsqueeze(0), self.attn(torch.cat(append, encoder_outputs[:, b]), 1) )
            attn_weights[b] = scores

        attn_weights = F.softmax(attn_weights)

        # input to bmm:
        # weights size: (batch size, 1, seq_len)
        # hidden states size: (seq_len, batch, hidden_size * num_directions)
        # transpose hidden state size: (batch, seq len, hidden_size * num_directions)
        # output size: (batch size, 1, hidden_size * num_directions)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0,1))

        # calculate 
        output = torch.cat((input, context.squeeze(1)), 1)
        output = self.out(output).unsqueeze(0)

        output = F.log_softmax(self.out(output[0]))
        return output, attn_weights