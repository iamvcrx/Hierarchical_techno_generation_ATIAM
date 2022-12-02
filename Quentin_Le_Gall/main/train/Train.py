from torch import nn
import torch
import os
from train.Spectral_Loss_Complex import Spectral_Loss


class train_VAE(nn.Module):
    def __init__(self, train_loader, model, writer, latent_dim,w, lr,n_fft_l, beta, model_name, num_epochs, save_ckpt,path_main): #, add_loss, add_figure
        super(train_VAE, self).__init__()

        self.w = w
        self.n_fft_l = n_fft_l
        self.beta = beta
        self.train_loader = train_loader
        self.model = model
        self.lr = lr
        self.path_main = path_main
        self.trained_model_path = f"{self.path_main}/trained_models/{model_name}.pt"
        self.writer = writer
        self.num_epochs = num_epochs
        self.latent_dim = latent_dim
        #self.add_loss = add_loss
        #self.add_figure = add_figure
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
            print(epoch)
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
                recons_loss += recons_loss_add

            # add loss in tensorboard 
            
            print(f"Epoch: {epoch+1} Loss tot.: {loss}, kl : {torch.abs(kl_div)}")
            self.writer.add_scalar("Loss/KL_div", torch.abs(kl_div), epoch)
            self.writer.add_scalar("Loss/Spectral_Loss", recons_loss, epoch)
            self.writer.add_scalar("Loss/Loss", loss, epoch)
            self.writer.flush()
            # save wheckpoint
            if epoch%5 == self.save_ckpt:
                # Save checkpoint if the model (to prevent training problem)
                checkpoint = {
                    "epoch": epoch + 1,
                    "VAE_model" : self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, self.trained_model_path)
            """ if n == self.add_loss:
                print(f"Epoch: {epoch} Loss tot.: {loss}, kl : {kl_div}")
                self.writer.add_scalar("Loss/KL_div", kl_div, epoch)
                self.writer.add_scalar("Loss/Spectral_Loss", recons_loss, epoch)
                self.writer.add_scalar("Loss/Loss", loss, epoch)
                self.writer.flush() """
                    
            """ # add generated pictures in tensorboard
            if n == self.add_figure:
                latent_space_samples = torch.randn(batch_size, self.latent_dim).to(device)
                generated_samples = self.generator(
                    latent_space_samples, mnist_labels
                )
                generated_samples = generated_samples.cpu().detach()

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

                

