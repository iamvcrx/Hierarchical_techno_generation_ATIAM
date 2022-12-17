from torch import nn


class Encoder_hierarchical(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 ratios: list = [4,4,2],
                 channel_size: list = [256,512,1024]
                 ):
        super(Encoder_hierarchical, self).__init__()

        self.latent_dim = latent_dim
        self.ratios = ratios
        self.channel_size = channel_size



        modules = nn.Sequential()
        # Build Encoder
        for i,ratio in enumerate(self.ratios):
            modules.append(
                    nn.Conv1d(in_channels, 
                              out_channels=self.channel_size[i],
                              kernel_size= (2*ratio)+1, 
                              stride = ratio, 
                              padding  = ratio))
            modules.append(
                    nn.BatchNorm1d(self.channel_size[i]))
            modules.append(
                    nn.LeakyReLU())
            
            in_channels = self.channel_size[i]

        self.encoder = modules
        
        self.fc_mu = nn.Sequential(
            nn.Conv1d(self.channel_size[-1], self.latent_dim,kernel_size = 9,stride=1,padding=4)
            )

        self.fc_var = nn.Sequential(
            nn.Conv1d(self.channel_size[-1], self.latent_dim,kernel_size = 9,stride=1,padding=4), 
            nn.Softplus()
            )

    def forward(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var