
class GAN_model(nn.Module):
    def __init__(self, e_input_size, e_hidden_size, d_input_size, d_hidden_size, output_size):
        self.G = G(e_input_size, e_hidden_size)
        self.D = D(d_input_size, d_hidden_size, output_size, self.encoder)

    def train(self):
        pass

    def test(self):
        pass