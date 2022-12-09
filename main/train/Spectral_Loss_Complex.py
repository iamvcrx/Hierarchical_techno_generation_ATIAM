import torch
import torch.nn as nn


def Spectral_Loss(x,y,n_fft_l=[2048,1024,512,256],w="Hamming",loss = "MSE",device='cpu'):
    """Calcul la loss spectrake moyennée sur les différentes valeurs de n_fft

    Inputs :
    - x : Batch
    - y : Batch
    - n_fft_l : Liste de n_fft
    - w : window type ("Hamming" or "Hanning")
    - device : device sur lequel effectuer ces calculs

    Outputs :
    - spectral_loss : tensor -> Moyenne des loss sur chaque fenêtre
    
    """
    n_sig = x.shape[-1]
    batch_size = x.shape[0]
    spectral_loss_tot = torch.zeros([len(n_fft_l)],dtype=torch.float64,device=device)

    for i,n_fft in enumerate(n_fft_l):
        #print(n_fft)
        n_win= n_fft
        n_hop = int(n_win*0.25)
        #L = int((n_sig - n_win + n_hop)/n_hop)
        #if n_fft%2 == 0 : #si pair
        #    I = int(n_fft/2 + 1)
        #else :
        #    I = int((n_fft-1)/2+1)
            

        if w=="Hamming":
            window = torch.hamming_window(n_win).to(device)
        if w=="Hanning":
            window = torch.hann_window(n_win).to(device)


        #lX = torch.zeros([batch_size,I, L], dtype=torch.float64,device=device)
        #lY = torch.zeros([batch_size,I, L], dtype=torch.float64,device=device)

        #for j, x_sample in enumerate(x) :
        #    X_xample = torch.stft(x_sample, n_fft, win_length=n_win, window=window,hop_length=n_hop,return_complex=True,center=False) #, center=True, pad_mode='reflect', normalized=False, onesided=None, return_complex=None
        #    lX_sample = torch.log(torch.abs(X_xample)**2 + 1)
        #    lX[j]=lX_sample
    
        #for j, y_sample in enumerate(y):
        #    Y_xample = torch.stft(y_sample, n_fft, win_length=n_win, window=window,hop_length=n_hop,return_complex=True,center=False) #, center=True, pad_mode='reflect', normalized=False, onesided=None, return_complex=None
        #    lY_sample = torch.log(torch.abs(Y_xample)**2 + 1)
        #    lY[j]=lY_sample

        lX = torch.stft(x, n_fft, win_length=n_win, window=window,hop_length=n_hop,return_complex=True,center=False) #, center=True, pad_mode='reflect', normalized=False, onesided=None, return_complex=None
        lY = torch.stft(y, n_fft, win_length=n_win, window=window,hop_length=n_hop,return_complex=True,center=False)
        lX = torch.log(torch.abs(lX) ** 2 + 1)
        lY = torch.log(torch.abs(lY) ** 2 + 1)

        if loss == "MSE":
            recons_criterion = nn.MSELoss(reduction="none")
            spectral_loss = recons_criterion(lX,lY).sum(1).mean()

        if loss == "L1":
            recons_criterion = nn.L1Loss(reduction="none")
            spectral_loss = recons_criterion(lX,lY).sum(1).mean()

        if loss == "MSE + L1":
            recons_criterion_MSE = nn.MSELoss(reduction="none")
            recons_criterion_L1 = nn.L1Loss(reduction="none")
            spectral_loss = recons_criterion_MSE(lX,lY).sum(1).mean() + recons_criterion_L1(lX,lY).sum(1).mean()
        #print(spectral_loss)
        spectral_loss_tot[i] = spectral_loss

    return spectral_loss_tot.mean()

