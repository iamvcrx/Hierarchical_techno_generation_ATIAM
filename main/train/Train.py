from torch import nn
import torch
import os
from train.Spectral_Loss_Complex import Spectral_Loss
import matplotlib.pyplot as plt
import numpy as np
import librosa 
import librosa.display

class train_VAE(nn.Module):
    def __init__(self, train_loader, model, writer, latent_dim,w, lr,n_fft_l, beta, model_name, num_epochs, save_ckpt,path_main,add_figure_sound): #, add_loss, add_figure
        super(train_VAE, self).__init__()

        self.w = w
        self.n_fft_l = n_fft_l
        self.beta = beta
        self.train_loader = train_loader
        self.model = model
        self.lr = lr
        self.path_main = path_main
        self.trained_model_path = "{}/trained_models/{}.pt".format(self.path_main, model_name)
        self.writer = writer
        self.num_epochs = num_epochs
        self.latent_dim = latent_dim
        #self.add_loss = add_loss
        self.add_figure_sound = add_figure_sound
        self.save_ckpt = save_ckpt



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

    def compute_loss(self,x,device): 
        y_pred,kl_div = self.model(x)
        recons_loss = Spectral_Loss(y_pred,x,n_fft_l=self.n_fft_l,w=self.w,device=device)
        full_loss = recons_loss - kl_div*self.beta
        return kl_div,recons_loss,full_loss


    def train_step(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        optimizer, start_epoch = self.load_checkpoint()
        print("Optimizer, ok")

        for epoch in range(start_epoch, start_epoch + self.num_epochs):
            
            loss = torch.Tensor([0]).to(device)
            kl_div = torch.Tensor([0]).to(device)
            recons_loss = torch.Tensor([0]).to(device)
            for n, x in enumerate(self.train_loader):
                x = x.to(device)

                # Compute the loss.
                kl_div_add,recons_loss_add,loss_add = self.compute_loss(x, device)
                # Before the backward pass, zero all of the network gradients
                optimizer.zero_grad()
                # Backward pass: compute gradient of the loss with respect to parameters
                loss_add.backward()
                # Calling the step function to update the parameters
                optimizer.step()
                #return loss
                loss += loss_add
                kl_div += kl_div_add
                recons_loss += recons_loss_add # diviser par le nombre de batch
            loss = loss/len(self.train_loader)
            kl_div = kl_div/len(self.train_loader)
            recons_loss = recons_loss/len(self.train_loader)
            # add loss in tensorboard 
            
            print("Epoch : {}, Loss tot : {}, kl : {}".format(epoch+1, loss, torch.abs(kl_div))) #f"Epoch: {epoch+1} Loss tot.: {loss}, kl : {torch.abs(kl_div)}"
            self.writer.add_scalar("Loss/KL_div", torch.abs(kl_div), epoch)
            self.writer.add_scalar("Loss/Spectral_Loss", recons_loss, epoch)
            self.writer.add_scalar("Loss/Loss", loss, epoch)
            self.writer.flush()
            # save wheckpoint
            if epoch%self.save_ckpt == 0:
                # Save checkpoint if the model (to prevent training problem)
                checkpoint = {
                    "epoch": epoch + 1,
                    "VAE_model" : self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, self.trained_model_path)


            if epoch%self.add_figure_sound == 0:
                # Save checkpoint if the model (to prevent training problem)
                nb_images = 2
                for i,batch_test in enumerate(self.train_loader):
                    if i>0:
                        break
                batch_test = batch_test.to(device)
                predictions,_ = self.model(batch_test)
                samples_rec = predictions[0:nb_images,:].cpu().detach().numpy()
                samples = batch_test[0:nb_images,:].cpu().detach().numpy()
                Fe = len(samples_rec[0][0])//2
                temps = np.arange(len(samples_rec[0][0]))/Fe

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

                self.writer.flush()


            """ if n == self.add_loss:
                print(f"Epoch: {epoch} Loss tot.: {loss}, kl : {kl_div}")
                self.writer.add_scalar("Loss/KL_div", kl_div, epoch)
                self.writer.add_scalar("Loss/Spectral_Loss", recons_loss, epoch)
                self.writer.add_scalar("Loss/Loss", loss, epoch)
                self.writer.flush()
                    

                figure = plt.figure()
                for i in range(16):
                    ax = plt.subplot(4, 4, i+1)
                    plt.imshow(generated_samples[i].reshape(28, 28), cmap='gray_r')
                    plt.title("label: " + str(mnist_labels[i].cpu().detach().numpy()))
                    plt.xticks([])
                    plt.yticks([])
                plt.tight_layout()

                self.writer.add_figure("4_mnist_images", figure, epoch)
                self.writer.flush()
                """

                

