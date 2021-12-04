import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Unflatten(nn.Module):
    def __init__(self, channel):
        super(Unflatten, self).__init__()
        self.channel = channel
    
    def forward(self, input):
        return input.view(self.channel,64,49)


class ConvVAE(nn.Module):
	def __init__(self, input_channel, input_dim,z_dim):
		super(ConvVAE, self).__init__()
		self.z_dim = z_dim
		self.input_dim = input_dim
		self.input_channel = input_channel

		self.encoder = nn.Sequential(
			nn.Conv1d(self.input_channel, 128, kernel_size=4,stride=2), #(200-4)/2+1=99
			nn.ReLU(),
			nn.Conv1d(128,64,kernel_size=3,stride=2), #(99-3)/2 +1=49
			nn.ReLU(),
			Flatten(), #49*64=
			nn.Linear(3136,1024),
			nn.ReLU(),
		)
		# hidden => mu
		self.fc21 = nn.Linear(1024,self.z_dim)
		# hidden => logvar
		self.fc22 =	nn.Linear(1024,self.z_dim)
  
		self.decoder = nn.Sequential(
			nn.Linear(self.z_dim,1024),
			nn.ReLU(),
			nn.Linear(1024,3136),
			nn.ReLU(),
			Unflatten(1),
			nn.ReLU(),
			nn.ConvTranspose1d(64, 128, kernel_size=3,stride=2),
   			nn.ReLU(),
			nn.ConvTranspose1d(128, self.input_channel, kernel_size=4, stride=2),
			nn.Tanh()
		)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)  # return tensor with same size as input filled with N(0, 1)
		return eps.mul(std).add_(mu) #mu + std * eps


	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar

	def encode(self,x):
		h = self.encoder(x)
		mu = self.fc21(h)
		logvar = self.fc22(h)
		return mu, logvar
  
	def decode(self,z):
		z = self.decoder(z)
		return z


    
