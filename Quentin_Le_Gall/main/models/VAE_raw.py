import torch
import torch.nn as nn




class VAE_raw(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(VAE_raw, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    #def encode(self,x):
    #    return self.encoder(x)

    #def decode(self, z):
    #    return self.decoder(z)
    def latent(self, z_params):
        
        mu, sigma = z_params
        std = torch.sqrt(sigma)
        eps = torch.rand_like(std)
        z = mu + eps * std

        kl_div = 0.5 * (1 + torch.log(sigma + 1e-6) - mu**2 - sigma).sum(1).mean()
        return z, kl_div


    def forward(self, x):
        # Encode the inputs
        #z_params = self.encode(x)
        z_params = self.encoder(x)

        # Obtain latent samples and latent loss
        z_tilde, kl_div = self.latent(z_params)

        # Decode the samples
        #x_tilde = self.decode(z_tilde)
        x_tilde = self.decoder(z_tilde)
        
        return x_tilde, kl_div
    
    
