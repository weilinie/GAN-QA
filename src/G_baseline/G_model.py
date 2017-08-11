import sys
import os
import random
sys.path.append(os.path.abspath(__file__ + "/../../"))
from model_zoo import *
import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


class G(nn.Module):
    def __init__(self, enc_input_size, enc_hidden_size, enc_n_layers, enc_num_directions,
                 dec_input_size, dec_hidden_size, output_size, dec_n_layers, dec_num_directions,
                 batch_size):
        super(G, self).__init__()
        self.encoder = EncoderRNN(enc_input_size, enc_hidden_size, batch_size, enc_n_layers, enc_num_directions)
        self.decoder = AttnDecoderRNN(dec_input_size, dec_hidden_size, output_size, self.encoder,
                                      dec_n_layers=1, dec_num_directions=1)


    def forward(self, inputs_ca, inputs_q, seq_lens, batch_size, max_q_len,
                embeddings_index, embeddings_size, word2index, index2word, teacher_forcing_ratio):
        # context encoding
        # output size: (seq_len, batch, hidden_size)
        # hidden size: (num_layers, batch, hidden_size)
        # the collection of all hidden states per batch is of size (seq_len, batch, hidden_size * num_directions)
        encoder_hiddens, encoder_hidden = self.encoder(inputs, seq_lens, None)

        # decoder
        # prepare decoder inputs as word embeddings in a batch
        # decoder_input size: (1, batch size, embedding size); first dim is 1 because only one time step;
        # nee to have a 3D tensor for input to nn.GRU module
        decoder_input = Variable(embeddings_index['SOS'].repeat(batch_size, 1).unsqueeze(0))
        if use_cuda:
            decoder_input = decoder_input.cuda()

        # use teacher forcing to step through each token in the decoder sequence
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(max_q_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, encoder_hiddens, embeddings_index)

                # accumulate loss
                targets = Variable(inputs_q[di].cuda()) if use_cuda else Variable(inputs_q[di])
                # loss += criterion(decoder_output, targets)

                # change next time step input to current target output, in embedding format
                decoder_input = Variable(torch.FloatTensor(1, batch_size, embeddings_size).cuda()) if use_cuda else \
                    Variable(torch.FloatTensor(1, batch_size, embeddings_size))
                for b in range(batch_size):
                    decoder_input[0, b] = embeddings_index[index2word[inputs_q[di, b]]].cuda() \
                        if use_cuda else \
                        embeddings_index[index2word[inputs_q[di, b]]]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(max_q_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, encoder_hiddens, embeddings_index)

                # top value and index of every batch
                # size of both topv, topi = (batch size, 1)
                topv, topi = decoder_output.data.topk(1)

                # get the output word for every batch
                decoder_input = Variable(torch.FloatTensor(1, batch_size, embeddings_size).cuda()) if use_cuda else \
                    Variable(torch.FloatTensor(1, batch_size, embeddings_size))
                for b in range(batch_size):
                    decoder_input[0, b] = embeddings_index[index2word[topi[0][0]]].cuda() if use_cuda else \
                        embeddings_index[index2word[topi[0][0]]]

                # accumulate loss
                # FIXME: in this batch version decoder, loss is accumulated for all <EOS> symbols even if
                # FIXME: the sentence has already ended. not sure if this is the right thing to do
                targets = Variable(inputs_q[di].cuda()) if use_cuda else Variable(inputs_q[di])
                # loss += criterion(decoder_output, targets)

        return decoder_output, targets


    def backward(self, out, labels, criterion, optimizer):
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        return loss