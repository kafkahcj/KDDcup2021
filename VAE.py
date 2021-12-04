
import numpy as np
import torch
import torch.nn as  nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

def loss_function_vae(recon_x, x, mu,logvar):
    #r_loss_function = nn.MSELoss()
    #reconstruct_loss = r_loss_function(recon_x, x)
    bce_loss =nn.BCELoss(reduction='mean')
    reconstruct_loss = bce_loss(recon_x, x)
    # KL Divergence to represent loss between distribution P and Q -> Decoder error
    # use logvar: latent log variance
    KLD = 0.5 * torch.sum(torch.exp(logvar) + torch.pow(mu, 2) - 1. - logvar)
    
    loss = KLD+reconstruct_loss
    return loss,reconstruct_loss,KLD


class VAE(nn.Module):
    
    def __init__(self, input_dim, z_dim):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.z_dim = z_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # hidden => mu
        self.fc31 = nn.Linear(64, self.z_dim)
        # hidden => logvar
        self.fc32 = nn.Linear(64, self.z_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim),
            nn.Sigmoid()
        )
        
    def encode(self,x):
        h = self.encoder(x)
        mu = self.fc31(h)
        logvar = self.fc32(h)
        return mu, logvar
    
    def decode(self,z):
        """
		Given a sampled z, decode it back to image
  		"""
        z = self.decoder(z)
        return z

    def reparametrize(self, mu, logvar):
	
 		#Given a standard gaussian distribution epsilon ~ N(0,1),
		#we can sample the random variable z as per z = mu + sigma * epsilon

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # return tensor with same size as input filled with N(0, 1)
        
        return eps.mul(std).add_(mu) #mu + std * eps
        
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        
        return res, mu, logvar
