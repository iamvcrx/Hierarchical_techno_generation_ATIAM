from torch import nn


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 ratios: list = [4,4,4,4],
                 channel_size: list = [16,32,64,128]
                 ):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.ratios = ratios
        self.channel_size = channel_size



        modules = nn.Sequential()
        self.decoder_input = nn.Conv1d(channel_size[-1],
                                       channel_size[-1],
                                       kernel_size=9,
                                       stride=1,
                                       padding=4)

        channel_size.reverse()
        ratios.reverse()
        # Build Encoder
        for i in range(len(channel_size) - 1):
            modules.append(
                    nn.ConvTranspose1d(channel_size[i],
                                       channel_size[i + 1],
                                       kernel_size=(ratios[i]*2)+1,
                                       stride = ratios[i],
                                       padding=ratios[i]))
            modules.append(
                    nn.LeakyReLU())
            modules.append(
                    nn.Conv1d(channel_size[i + 1],
                              channel_size[i + 1],
                              kernel_size=(ratios[i]*2)+1,
                              stride = 1,
                              padding=ratios[i]))
            modules.append(
                    nn.BatchNorm1d(channel_size[i + 1]))
            modules.append(
                    nn.LeakyReLU())
            
        self.decoder = modules

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose1d(channel_size[-1],
                                               in_channels,
                                               kernel_size=(2*ratios[-1])+1,
                                               stride=ratios[-1],
                                               padding=ratios[-1]),
                            nn.BatchNorm1d(channel_size[-1]),
                            nn.LeakyReLU(),
                            nn.Conv1d(in_channels, 
                                      in_channels,
                                      kernel_size= (2*ratios[-1])+1,
                                      stride = 1,
                                      padding = ratios[-1]),
                            nn.Tanh())


    def forward(self, input):
        result = self.decoder_input(input)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result