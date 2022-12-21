import torch
import torch.nn as nn

def KL_Loss(_mu1,_sigma1,_mu2,_sigma2,type = "forward"):
    """Calcul la loss KL entre deux distributions

    Inputs :
    - mu1 : Batch
    - mu2 : Batch
    - sigma1 : Batch
    - sigma1 : Batch
    - device : device sur lequel effectuer ces calculs

    Outputs :
    - KL_Loss : tensor -> loss of two distributions
    
    """
    if type == "forward":
        kl_loss = 1/2 * torch.log(_sigma2/_sigma1 + 1e-6) + (torch.square(_sigma1) + torch.square(_mu1 - _mu2))/(2*torch.square(_sigma2)) -1/2
    if type == "backward":
        kl_loss = 1/2 * torch.log(_sigma2/_sigma1 + 1e-6) + (torch.square(_sigma1) + torch.square(_mu1 - _mu2))/(2*torch.square(_sigma2)) -1/2
    if type == "mean":
        kl_loss_1 = 1/2 * torch.log(_sigma2/_sigma1 + 1e-6) + (torch.square(_sigma1) + torch.square(_mu1 - _mu2))/(2*torch.square(_sigma2)) -1/2
        kl_loss_2 = 1/2 * torch.log(_sigma1/_sigma2 + 1e-6) + (torch.square(_sigma2) + torch.square(_mu2 - _mu1))/(2*torch.square(_sigma1)) -1/2
        kl_loss = (kl_loss_1 + kl_loss_2)/2
    #print(kl_loss.shape)
    return kl_loss.sum(dim=1).mean()