import torch
import torch.nn as nn


def Spectral_Loss(x,
                  y,
                  n_fft_l = [2048,1024,512,256],
                  w = "Hamming",
                  loss = "MSE",
                  device = 'cpu'):
    """Calcul la loss spectrale moyennée sur les différentes valeurs de n_fft

    Inputs :
    - x : Batch
    - y : Batch
    - n_fft_l : Liste de n_fft
    - w : window type ("Hamming" or "Hanning")
    - device : device sur lequel effectuer ces calculs

    Outputs :
    - spectral_loss : tensor -> Moyenne des loss sur chaque fenêtre
    
    """
    # On enlève la dimension inutile pour que la stft puisse se faire
    x = x[:,0,:]
    y = y[:,0,:]
    spectral_loss_tot = torch.zeros([len(n_fft_l)],dtype=torch.float64,device=device)

    for i,n_fft in enumerate(n_fft_l):

        n_win= n_fft
        n_hop = int(n_win*0.25)

            
        # Window pour la stft
        if w=="Hamming":
            window = torch.hamming_window(n_win).to(device)
        if w=="Hanning":
            window = torch.hann_window(n_win).to(device)

        # STFT et log
        X = torch.stft(x, n_fft, win_length=n_win, window=window,hop_length=n_hop,return_complex=True,center=False)
        X = torch.log(torch.abs(X)**2 + 1)
        
        Y = torch.stft(y, n_fft, win_length=n_win, window=window,hop_length=n_hop,return_complex=True,center=False)
        Y = torch.log(torch.abs(Y)**2 + 1)

        if loss == "MSE":
            recons_criterion = nn.MSELoss(reduction="none")
            spectral_loss = recons_criterion(X,Y).mean()

        if loss == "L1":
            recons_criterion = nn.L1Loss(reduction="none")
            spectral_loss = recons_criterion(X,Y).mean()
            
        if loss == "MSE_L1":
            recons_criterion_MSE = nn.MSELoss(reduction="none")
            recons_criterion_L1 = nn.L1Loss(reduction="none")
            spectral_loss = recons_criterion_MSE(X,Y).mean() + recons_criterion_L1(X,Y).mean()

        
        spectral_loss_tot[i] = spectral_loss

    return spectral_loss_tot.mean()
