from torch import nn
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import librosa 
import librosa.display
import train.KL_Loss

class train_VAE_hierarchical(nn.Module):
    def __init__(self, 
                 train_loader_latent,
                 valid_loader_latent,  
                 model_raw,
                 model_hierarchical,
                 run_name,

                 latent_dim, 
                 lr, 
                 beta, 
                 model_name, 
                 num_epochs, 
                 
                 writer, 
                 save_ckpt,
                 path_trained_model,
                 add_figure_sound,
                 device,
                 valid_loader_raw): #, add_loss, add_figure
        super(train_VAE_hierarchical, self).__init__()

        
        self.train_loader = train_loader_latent
        self.valid_loader = valid_loader_latent
        self.model_raw = model_raw
        self.model_hierarchical = model_hierarchical

        self.valid_loader_raw = valid_loader_raw

        self.lr = lr
        self.beta = beta
        self.run_name = run_name
        self.path_trained_model = path_trained_model
        self.trained_model_path = "{}/{}/{}/{}.pt".format(self.path_trained_model, run_name, model_name, model_name)
        self.best_trained_model_path = "{}/{}/{}/{}_best.pt".format(self.path_trained_model, run_name, model_name, model_name)
        self.writer = writer
        self.num_epochs = num_epochs
        self.latent_dim = latent_dim
        self.add_figure_sound = add_figure_sound
        self.save_ckpt = save_ckpt
        self.device = device


    def load_checkpoint(self):
        optimizer = self.configure_optimizer()
        if os.path.isfile(self.trained_model_path):
            ckpt = torch.load(self.trained_model_path)

            self.model_hierarchical.load_state_dict(ckpt["VAE_model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"]
            print("model parameters loaded from " + self.trained_model_path)
        else:
            start_epoch = 0
            print("new model")
            
        return optimizer, start_epoch


    def configure_optimizer(self):
        optimizer = torch.optim.Adam(
            self.model_hierarchical.parameters(), lr=self.lr
        )
 
        return optimizer

    def compute_loss(self,x): 
        y_pred,kl_div = self.model_hierarchical(x)
        mu2,sigma2 = torch.split(y_pred,128,dim = 1)
        mu1,sigma1 = torch.split(x,128,dim = 1)
        recons_loss = train.KL_Loss.KL_Loss(mu1,sigma1,mu2,sigma2)
        full_loss = recons_loss - kl_div*self.beta
        return kl_div,recons_loss,full_loss


    def train_step(self):
        optimizer, start_epoch = self.load_checkpoint()
        print("Optimizer, ok")
        
        for epoch in range(start_epoch, start_epoch + self.num_epochs):
                
            loss = torch.Tensor([0]).to(self.device)
            kl_div = torch.Tensor([0]).to(self.device)
            recons_loss = torch.Tensor([0]).to(self.device)
            
            for n, x in enumerate(self.train_loader):
                
                x = x[0].to(self.device)
                # Compute the loss.
                kl_div_add,recons_loss_add,loss_add = self.compute_loss(x)
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
            
            print(f"Epoch: {epoch+1} Loss tot.: {loss}, kl : {torch.abs(kl_div)}")
            self.writer.add_scalar("Loss_Hierarchical/KL_div", torch.abs(kl_div), epoch)
            self.writer.add_scalar("Loss_Hierarchical/MSE", recons_loss, epoch)
            self.writer.add_scalar("Loss_Hierarchical/Loss_tot", loss, epoch)
            self.writer.flush()

############################ VALID LOSS ##############################
            valid_loss = torch.Tensor([0]).to(self.device)
            for n, x in enumerate(self.valid_loader):
                x = x[0].to(self.device)
                with torch.no_grad():
                    _,_, valid_loss_add = self.compute_loss(x)
                valid_loss += valid_loss_add
            valid_loss = valid_loss/len(self.valid_loader)
            self.writer.add_scalar("Loss_Hierarchical/Valid_Loss", valid_loss, epoch)

############################ STOPPING CRITERION ##############################
            if epoch==start_epoch:
                old_valid = valid_loss
                min_valid = valid_loss
                counter = 0
            if valid_loss < min_valid:
                min_valid = valid_loss
                counter = 0
                ############### Save best checkpoint ################
                checkpoint = {
                    "epoch": epoch + 1,
                    "VAE_model" : self.model_hierarchical.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, self.best_trained_model_path)
                ############### Save best checkpoint ################

            # if old_valid <= valid_loss:
            #     counter += 1

            # if counter >= 10 :
            #     print("Overfitting, train stopped")
            #     break

            old_valid = valid_loss



############################ SAVE CHECKPOINT #############################
            if epoch%self.save_ckpt == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "VAE_model" : self.model_hierarchical.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, self.trained_model_path) 
            

##################################### VISU ####################################
            # Figure 
            if epoch%self.add_figure_sound == 0:
                nb_images = 3
                batch_visu = next(iter(self.valid_loader))
                # for i, batch_visu in enumerate(self.valid_loader):
                #     if i>1:
                #         break
                batch_visu = batch_visu[0].to(self.device)

                # Construction audio sans model 2
                mu, sigma = torch.split(batch_visu,128,dim = 1)
                with torch.no_grad():
                    waveform_ori,_ = self.model_raw.latent((mu, sigma))
                    waveform_ori = self.model_raw.decoder(waveform_ori)




                """ with torch.no_grad() :
                mu_sigma_raw = model_hier.decoder(torch.from_numpy(z1))
                mu_raw, sigma_raw = torch.split(mu_sigma_raw, 128, dim=1)

                z_raw, _ = model_raw.latent((mu_raw, sigma_raw))
                xtilde = model_raw.decoder(z_raw)

                print(xtilde.shape) """

                # Avec les deu modèes
                recons,_ = self.model_hierarchical(batch_visu)
                mu_rec, sigma_rec = torch.split(recons,128,dim = 1)
                with torch.no_grad():
                    waveform_rec,_ = self.model_raw.latent((mu_rec, sigma_rec))
                    waveform_rec = self.model_raw.decoder(waveform_rec)
                    waveform_rec_mu = self.model_raw.decoder(mu_rec)
                
                ori_samples = waveform_ori[0:nb_images,:].cpu().detach().numpy()
                rec_samples = waveform_rec[0:nb_images,:].cpu().detach().numpy()
                rec_samples_mu = waveform_rec_mu[0:nb_images,:].cpu().detach().numpy()
                Fe = len(rec_samples[0][0])//2



                # Plot
                figure, ax = plt.subplots(nrows=nb_images, sharex=True)
                for i in range(nb_images):
                    # ax[i].plot(ori_samples[i], alpha=0.9, label='Original')
                    # ax[i].plot(rec_samples[i],color='r', alpha=0.5, label='Reconstructed')
                    librosa.display.waveshow(ori_samples[i][0,:], sr=Fe, alpha=0.9, ax=ax[i], label='VAE 1')
                    librosa.display.waveshow(rec_samples[i][0,:], sr=Fe, color='r', alpha=0.5, ax=ax[i], label='VAE 1 + VAE 2')
                    librosa.display.waveshow(rec_samples_mu[i][0,:], sr=Fe, color='orange', alpha=0.5, ax=ax[i], label='VAE 1 + mu VAE 2')
                    ax[i].set(title=f'Latent {i+1}')
                    #plt.grid()
                    if i==0:
                        ax[i].legend()

                plt.tight_layout()

                
                self.writer.add_figure("Waveform", figure, epoch)
                
                origin = np.array(ori_samples[0][0,:]/np.max(np.abs(ori_samples[0][0,:])),dtype=np.float32)
                reconstructed = np.array(rec_samples[0][0,:]/np.max(np.abs(rec_samples[0][0,:])),dtype=np.float32)
                reconstructed_mu = np.array(rec_samples_mu[0][0,:]/np.max(np.abs(rec_samples_mu[0][0,:])),dtype=np.float32)
                self.writer.add_audio("Sound_hierarchical/Sound_1/Only VAE 1", origin, epoch,sample_rate=Fe)
                self.writer.add_audio("Sound_hierarchical/Sound_1/VAE 1 + VAE 2", reconstructed, epoch,sample_rate=Fe)
                self.writer.add_audio("Sound_hierarchical/Sound_1/VAE 1 + mu VAE 2", reconstructed_mu, epoch,sample_rate=Fe)

                origin1 = np.array(ori_samples[1][0,:]/np.max(np.abs(ori_samples[1][0,:])),dtype=np.float32)
                reconstructed1 = np.array(rec_samples[1][0,:]/np.max(np.abs(rec_samples[1][0,:])),dtype=np.float32)
                reconstructed_mu1 = np.array(rec_samples_mu[1][0,:]/np.max(np.abs(rec_samples_mu[1][0,:])),dtype=np.float32)
                self.writer.add_audio("Sound_hierarchical/Sound_2/Only VAE 1", origin1, epoch,sample_rate=Fe)
                self.writer.add_audio("Sound_hierarchical/Sound_2/VAE 1 + VAE 2", reconstructed1, epoch,sample_rate=Fe)
                self.writer.add_audio("Sound_hierarchical/Sound_2/VAE 1 + mu VAE 2", reconstructed_mu1, epoch,sample_rate=Fe)

                origin2 = np.array(ori_samples[2][0,:]/np.max(np.abs(ori_samples[2][0,:])),dtype=np.float32)
                reconstructed2 = np.array(rec_samples[2][0,:]/np.max(np.abs(rec_samples[2][0,:])),dtype=np.float32)
                reconstructed_mu2 = np.array(rec_samples_mu[2][0,:]/np.max(np.abs(rec_samples_mu[2][0,:])),dtype=np.float32)
                self.writer.add_audio("Sound_hierarchical/Sound_3/Only VAE 1", origin2, epoch,sample_rate=Fe)
                self.writer.add_audio("Sound_hierarchical/Sound_3/VAE 1 + VAE 2", reconstructed2, epoch,sample_rate=Fe)
                self.writer.add_audio("Sound_hierarchical/Sound_3/VAE 1 + mu VAE 2", reconstructed_mu2, epoch,sample_rate=Fe)

                self.writer.flush()

            """
            if epoch%self.add_figure_sound == 0:
                # Save checkpoint if the model (to prevent training problem)
                nb_images = 2
                for i,batch_test in enumerate(self.train_loader):
                    if i>0:
                        break
                batch_test = batch_test.to(self.device)
                z_params_test = self.model_raw.encode(batch_test)
                z_test, kl_div_test = self.model_raw.latent(z_params_test)
                predictions,_ = self.model_hierarchical(z_test)
                samples_rec = predictions[0:nb_images,:].cpu().detach().numpy()
                samples = z_test[0:nb_images,:].cpu().detach().numpy()
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
                #plt.show()

                figure = plt.figure()
                
                    ax = plt.subplot(nb_images, nb_images, i+1)
                    plt.plot(temps, samples[i][0,:],label = "Origin")
                    plt.plot(temps, samples_rec[i][0,:],label = "Reconstruct")
                    plt.title(f"Son {i+1}")
                    plt.xlabel("time (s)")
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

            
            if n == self.add_loss:
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
        #torch.save(checkpoint, self.trained_model_path) # Tu peut enlever ça je pense