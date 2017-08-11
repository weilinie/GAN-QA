import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../")
from model_zoo import *


class G(nn.Module):
    def __init__(self, e_input_size, e_hidden_size, d_input_size, d_hidden_size, output_size):
        super(G, self).__init__()
        self.encoder = EncoderRNN(e_input_size, e_hidden_size)
        self.decoder = AttnDecoderRNN(d_input_size, d_hidden_size, output_size, self.encoder)

# TODAY: finish this