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
        self.in_channels = in_channels


        modules = nn.Sequential()
        self.decoder_input = nn.Conv1d(self.latent_dim,
                                       self.channel_size[-1],
                                       kernel_size=9,
                                       stride=1,
                                       padding=4)

        self.channel_size.reverse()
        self.ratios.reverse()
        # Build Encoder
        for i in range(len(self.channel_size) - 1):
            modules.append(
                    nn.ConvTranspose1d(self.channel_size[i],
                                       self.channel_size[i + 1],
                                       kernel_size = (self.ratios[i]*2)+1,
                                       stride = self.ratios[i],
                                       padding = self.ratios[i],
                                       output_padding = self.ratios[i]-1))
            modules.append(
                    nn.LeakyReLU())
            modules.append(
                    nn.Conv1d(self.channel_size[i + 1],
                              self.channel_size[i + 1],
                              kernel_size=(self.ratios[i]*2)+1,
                              stride = 1,
                              padding=self.ratios[i]))
            modules.append(
                    nn.BatchNorm1d(self.channel_size[i + 1]))
            modules.append(
                    nn.LeakyReLU())
            
        self.decoder = modules
        
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose1d(self.channel_size[-1],
                                               self.in_channels,
                                               kernel_size=(2*self.ratios[-1])+1,
                                               stride=self.ratios[-1],
                                               padding=self.ratios[-1],
                                               output_padding = self.ratios[-1]-1),
                            nn.LeakyReLU(),
                            nn.Conv1d(self.in_channels, 
                                      self.in_channels,
                                      kernel_size= (2*self.ratios[-1])+1,
                                      stride = 1,
                                      padding = self.ratios[-1]),
                            nn.BatchNorm1d(self.in_channels),
                            nn.Tanh())

        
    def forward(self, input):
        result = self.decoder_input(input)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result