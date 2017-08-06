from model_zoo import *


class G(nn.Module):
    def __init__(self, e_input_size, e_hidden_size, d_input_size, d_hidden_size, output_size):
        self.encoder = EncoderRNN(e_input_size, e_hidden_size)
        self.decoder = DecoderRNN(d_input_size, d_hidden_size, output_size, self.encoder)