import torch
import torch.nn as nn
import numpy as np

def Spectral_Loss(x,y,n_fft=16384,win_lenght=1024,window="Hamming",epsilon = 1):
    if window=="Hamming":
        window = torch.hamming_window(win_lenght)
        lX = []
        lY = []
    for x_sample in x :
        X_xample = torch.stft(x_sample, n_fft, win_length=win_lenght, window=window,return_complex=True) #, center=True, pad_mode='reflect', normalized=False, onesided=None, return_complex=None
        lX_sample = torch.log(torch.abs(X_xample)**2 + epsilon)
        lX.append(lX_sample.detach().numpy())
    for y_sample in y:
        Y_xample = torch.stft(y_sample, n_fft, win_length=win_lenght, window=window,return_complex=True) #, center=True, pad_mode='reflect', normalized=False, onesided=None, return_complex=None
        lY_sample = torch.log(torch.abs(Y_xample)**2 + epsilon)
        lY.append(lY_sample.detach().numpy())



    lX = torch.tensor(np.asarray(lX))
    lY = torch.tensor(np.asarray(lY))
    recons_criterion = nn.L1Loss()
    spectral_loss = recons_criterion(lX,lY)

    return spectral_loss