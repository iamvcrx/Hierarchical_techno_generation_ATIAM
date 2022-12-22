from torch import nn
import torch
import os
from train.Spectral_Loss_Complex import Spectral_Loss
import matplotlib.pyplot as plt
import numpy as np
import librosa 
import librosa.display

class train_VAE(nn.Module):
    def __init__(self, 
                 train_loader, 
                 model, writer, 
                 latent_dim,
                 w, 
                 lr,
                 n_fft_l, 
                 beta, 
                 model_name, 
                 num_epochs, 
                 save_ckpt,
                 path_main,
                 add_figure_sound,
                 loss,
                 device,
                 valid_loader):
        super(train_VAE, self).__init__()

        self.w = w
        self.n_fft_l = n_fft_l
        self.beta = beta
        self.train_loader = train_loader
        self.model = model
        self.lr = lr
        self.path_main = path_main
        self.trained_model_path = "{}/runs/{}/{}.pt".format(self.path_main, model_name,model_name)
        self.best_trained_model_path = "{}/runs/{}/{}_best.pt".format(self.path_main, model_name,model_name)
        self.writer = writer
        self.num_epochs = num_epochs
        self.latent_dim = latent_dim
        #self.add_loss = add_loss
        self.add_figure_sound = add_figure_sound
        self.save_ckpt = save_ckpt
        self.loss = loss
        self.device = device
        self.valid_loader = valid_loader



    def load_checkpoint(self):
        optimizer = self.configure_optimizer()
        if os.path.isfile(self.trained_model_path):
            ckpt = torch.load(self.trained_model_path)

            self.model.load_state_dict(ckpt["VAE_model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"]
            print("model parameters loaded from " + self.trained_model_path)
        else:
            start_epoch = 0
            print("new model")
            
        return optimizer, start_epoch


    def configure_optimizer(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr
        )
 
        return optimizer

    def compute_loss(self,x): 
        y_pred,kl_div = self.model(x)
        recons_loss = Spectral_Loss(y_pred,x,n_fft_l=self.n_fft_l,w=self.w,loss = self.loss,device=self.device)
        full_loss = recons_loss - kl_div*self.beta
        return kl_div,recons_loss,full_loss


    def train_step(self):
        optimizer, start_epoch = self.load_checkpoint()
        print("Optimizer, ok")

        for epoch in range(start_epoch, start_epoch + self.num_epochs):
            
            ################## Training loop ####################
            loss = torch.Tensor([0]).to(self.device)
            kl_div = torch.Tensor([0]).to(self.device)          # Initialization
            recons_loss = torch.Tensor([0]).to(self.device)

            for n, x in enumerate(self.train_loader):
                x = x.to(self.device)
                # Compute the loss.
                kl_div_add,recons_loss_add,loss_add = self.compute_loss(x)

                # Before the backward pass, zero all of the network gradients
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to parameters
                loss_add.backward()

                # Calling the step function to update the parameters
                optimizer.step()

                # Somme des loss sur tous les batches
                loss += loss_add
                kl_div += kl_div_add
                recons_loss += recons_loss_add

            # Normalisation par le nombre de batch
            loss = loss/len(self.train_loader)
            kl_div = kl_div/len(self.train_loader)   
            recons_loss = recons_loss/len(self.train_loader)

            # Add loss in tensorboard 
            print("Epoch : {}, Loss tot : {}, kl : {}".format(epoch+1, loss, torch.abs(kl_div))) 
            self.writer.add_scalar("Loss/KL_div", torch.abs(kl_div), epoch) 
            self.writer.add_scalar("Loss/Spectral_Loss", recons_loss, epoch)
            self.writer.add_scalar("Loss/Loss", loss, epoch)
            self.writer.flush()

            # Save checkpoint
            if epoch%self.save_ckpt == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "VAE_model" : self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, self.trained_model_path)


            ##################### Valid ################################
            counter = torch.Tensor([0]).to(self.device)
            valid_loss = torch.Tensor([0]).to(self.device)
            
            for n, x in enumerate(self.valid_loader):
                x = x.to(self.device)
                with torch.no_grad():
                    _,_, valid_loss_add = self.compute_loss(x)
                valid_loss += valid_loss_add
            valid_loss = valid_loss/len(self.valid_loader)
            self.writer.add_scalar("Loss/Valid_Loss", valid_loss, epoch)

            # Stopping criterion
            if epoch==start_epoch:
                old_valid = valid_loss
                min_valid = valid_loss
            if valid_loss < min_valid:
                min_valid = valid_loss
                counter = 0
                # Save best checkpoint
                checkpoint = {
                    "epoch": epoch + 1,
                    "VAE_model" : self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, self.best_trained_model_path)

            if old_valid < valid_loss:
                counter += 1

            # if counter >= 10 :
            #     print("Overfitting, train stopped")
            #     break

            old_valid = valid_loss


            ##################### Visu #############################
            if epoch%self.add_figure_sound == 0:
                nb_images = 3
                batch_test = next(iter(self.valid_loader))
                
                batch_test = batch_test.to(self.device)
                predictions,_ = self.model(batch_test)
                samples_rec = predictions[0:nb_images,:].cpu().detach().numpy()
                samples = batch_test[0:nb_images,:].cpu().detach().numpy()
                Fe = len(samples_rec[0][0])//2

                figure, ax = plt.subplots(nrows=nb_images, sharex=True)
                for i in range(nb_images):
                    librosa.display.waveshow(samples[i][0,:], sr=Fe, alpha=0.9, ax=ax[i], label='Original')
                    librosa.display.waveshow(samples_rec[i][0,:], sr=Fe, color='r', alpha=0.5, ax=ax[i], label='Reconstructed')
                    ax[i].set(title=f'Sound {i+1}')
                    #plt.grid()
                    if i==0:
                        ax[i].legend()

                plt.tight_layout()

                self.writer.add_figure("Waveform", figure, epoch)

                origin = np.array(samples[0][0,:]/np.max(np.abs(samples[0][0,:])),dtype=np.float32)
                reconstructed = np.array(samples_rec[0][0,:]/np.max(np.abs(samples_rec[0][0,:])),dtype=np.float32)
                self.writer.add_audio("Sound_1/Original sound", origin, epoch,sample_rate=Fe)
                self.writer.add_audio("Sound_1/Reconstructed sound", reconstructed, epoch,sample_rate=Fe)

                origin1 = np.array(samples[1][0,:]/np.max(np.abs(samples[1][0,:])),dtype=np.float32)
                reconstructed1 = np.array(samples_rec[1][0,:]/np.max(np.abs(samples_rec[1][0,:])),dtype=np.float32)
                self.writer.add_audio("Sound_2/Original sound", origin1, epoch,sample_rate=Fe)
                self.writer.add_audio("Sound_2/Reconstructed sound", reconstructed1, epoch,sample_rate=Fe)

                origin2 = np.array(samples[2][0,:]/np.max(np.abs(samples[2][0,:])),dtype=np.float32)
                reconstructed2 = np.array(samples_rec[2][0,:]/np.max(np.abs(samples_rec[2][0,:])),dtype=np.float32)
                self.writer.add_audio("Sound_3/Original sound", origin2, epoch,sample_rate=Fe)
                self.writer.add_audio("Sound_3/Reconstructed sound", reconstructed2, epoch,sample_rate=Fe)

                self.writer.flush()

                

