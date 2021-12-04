import torch
import torch.nn as nn


class ConvAE(nn.Module):
	def __init__(self, input_channel, input_dim,z_dim):
		super(ConvAE, self).__init__()
		self.z_dim = z_dim
		self.input_dim = input_dim
		self.input_channel = input_channel
  
		self.encoder = nn.Sequential(
			nn.Conv1d(self.input_channel, 64, kernel_size=3,stride=1), 
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Conv1d(64,self.z_dim,kernel_size=4,stride=2), 
			nn.ReLU(),
		)


		self.decoder = nn.Sequential(
			nn.ConvTranspose1d(self.z_dim, 64, kernel_size=4,stride=2),
			nn.Dropout(0.1),
			nn.ReLU(),
			nn.ConvTranspose1d(64, input_channel, kernel_size=3, stride=1),
			nn.Tanh()
		)

	def forward(self, x):
		res = x
		res = self.encoder(res)
		res = self.decoder(res)
		return res
