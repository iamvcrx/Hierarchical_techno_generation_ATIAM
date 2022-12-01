import torch
import torch.nn as nn
import numpy as np

def Spectral_Loss(x, y, Nfft=512, Nwin=512, window="Hamming", Nhop = 512//4, epsilon = 1):
    Nsig = x.shape[2]
    batch_size = x.shape[0]

    L = int((Nsig - Nwin + Nhop)/Nhop)

    if Nfft % 2 == 0 : #si pair
        I = int(Nfft/2 + 1)
    else :
        I = int((Nfft-1)/2+1)
        
    if window=="Hamming":
        window = torch.hamming_window(Nwin)
    if window=="Hanning":
        window = torch.hann_window(Nwin)

    lX = torch.zeros([batch_size, I, L], dtype=torch.cfloat)
    lY = torch.zeros([batch_size, I, L], dtype=torch.cfloat)

    for i, x_sample in enumerate(x) :
        X_xample = torch.stft(x_sample, Nfft, win_length=Nwin, window=window, hop_length=Nhop, return_complex=True, center=False) #, center=True, pad_mode='reflect', normalized=False, onesided=None, return_complex=None
        lX_sample = torch.log(torch.abs(X_xample)**2 + epsilon)
        lX[i] = lX_sample

    for i, y_sample in enumerate(y):
        Y_xample = torch.stft(y_sample, Nfft, win_length=Nwin, window=window, hop_length=Nhop, return_complex=True, center=False) #, center=True, pad_mode='reflect', normalized=False, onesided=None, return_complex=None
        lY_sample = torch.log(torch.abs(Y_xample)**2 + epsilon)
        lY[i] = lY_sample


    recons_criterion = nn.L1Loss(reduction="none")
    spectral_loss = np.sum(np.mean(recons_criterion(lX,lY),axis=0))

    return spectral_loss