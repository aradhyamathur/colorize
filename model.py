import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from UNet import *
from UNet_short import *

class UNetEncDec(nn.Module):
	def __init__(self, path, out_channels=3):
		super(UNetEncDec, self).__init__()
		self.encoder = UnetShort(path)
		self.encoder.requires_grad = False
		self.decoder = ColorDecoderConvTrans(out_channels=out_channels)

	def forward(self, x):
		out = self.encoder(x)
		out = self.decoder(out)
		return out


class UNetEncoder(nn.Module):
	def __init__(self, encoder, out_channels=3):
		self.encoder = encoder
		self.encoder.requires_grad = False
		self.decoder = ColorDecoderConvTrans(out_channels=out_channels)

	def forward(self, x):
		out = self.encoder(x)
		out = self.decoder(out)
		return out

class Encoder(nn.Module):

	def __init__(self):

		super(Encoder, self).__init__()
		self.conv1 = nn.Conv2d(1, 128, 3, padding=1, stride=2)
		self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
		# self.conv3 = nn.Conv2d(512, 512, 3, padding=1)
		self.conv4 = nn.Conv2d(256, 512, 3, padding=1,stride=2)
		self.conv5 = nn.Conv2d(512, 1024, 3, padding=1, stride=2)  
		# self.conv6 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
		# self.conv7 = nn.Conv2d(256, 128, 3, padding=1, stride=2)
		self.bn1 = nn.BatchNorm2d(128)
		self.bn2 = nn.BatchNorm2d(256)
		# self.bn3 = nn.BatchNorm2d(512)
		self.bn4 = nn.BatchNorm2d(512)
		self.bn5 = nn.BatchNorm2d(1024)
		# self.bn6 = nn.BatchNorm2d(256)
		# self.bn7 = nn.BatchNorm2d(128)	



		for m in self.modules():
			if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
				print('Initializing', m)
				nn.init.xavier_normal_(m.weight)
	
	def forward(self, x):

		#print('ENCODER')

		out = self.bn1(F.leaky_relu(self.conv1(x)))

		# print('Conv1:', out.shape)

		out = self.bn2(F.leaky_relu(self.conv2(out)))

		# print('Conv2: ', out.shape)
		out = F.dropout2d(out, p=0.3, training=self.training)
		# out = self.bn3(F.leaky_relu(self.conv3(out)))

		# print('Conv3: ', out.shape)
		# out = F.dropout2d(out, p=0.3, training=self.training)
		# print(out.shape)
		out = self.bn4(F.leaky_relu(self.conv4(out)))
		# print('Conv4: ', out.shape)
		out = F.dropout2d(out, p=0.3, training=self.training)
		out = self.bn5(F.leaky_relu(self.conv5(out)))
		# print('Conv5: ', out.shape)
		# out = F.dropout2d(out, p=0.3, training=self.training)
		# out = self.bn6(F.leaky_relu(self.conv6(out)))
		# print('Conv6: ', out.shape)
		# out = F.dropout2d(out, p=0.3, training=self.training)
		# out = self.bn7(F.leaky_relu(self.conv7(out)))
		# print('Conv7: ', out.shape)
		return out


class ColorDecoderConvTrans(nn.Module):

	def __init__(self, out_channels=1):

		super(ColorDecoderConvTrans, self).__init__()
		# self.upsample1 = nn.Upsample(scale_factor=4)
		# self.upsample2 = nn.Upsample(scale_factor=2)

		self.conv1 = nn.ConvTranspose2d(1024, 512, 3, padding=1, stride=2, output_padding=1)
		# self.conv2 = nn.Conv2d(512, 512, 3, padding=1)
		self.conv3 = nn.ConvTranspose2d(512, 256, 3, padding=1, stride=2, output_padding=1)
		# self.conv4 = nn.Conv2d(512, 128, 3, padding=1)
		self.conv5 = nn.ConvTranspose2d(256, 256, 3, padding=1, stride=2, output_padding=1)
		# self.conv6 = nn.Conv2d(128,	 256, 3, padding=1)
		self.conv7 = nn.ConvTranspose2d(256, 3, 3, padding=1, stride=2, output_padding=1)
		# self.conv8 = nn.Conv2d(128, 3, 3, padding=1)

		self.bn1 = nn.BatchNorm2d(512)
		self.bn2 = nn.BatchNorm2d(256)
		# self.bn3 = nn.BatchNorm2d(512)
		self.bn4 = nn.BatchNorm2d(256)
		# self.bn5 = nn.BatchNorm2d(256)
		# self.bn6 = nn.BatchNorm2d(256)
		# self.bn7 = nn.BatchNorm2d(128)


		for m in self.modules():
			if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
				print('Initializing', m)
				nn.init.xavier_normal_(m.weight)

	def forward(self, x):
		# print(x.shape)
		#print('DECODER')
		out = self.bn1(F.leaky_relu(self.conv1(x)))
		#print('Conv1 : ', out.shape)

		# out = self.upsample1(out)
		out = F.dropout2d(out, p=0.3, training=self.training)
		out = self.bn2(F.leaky_relu(self.conv3(out)))
		#print('Conv2: ', out.shape)

		# out = self.upsample2(out)
		# out = F.dropout2d(out, p=0.3, training=self.training)
		# out = self.bn3(F.leaky_relu(self.conv3(out)))
		#print('Conv3: ', out.shape)
		out = F.dropout2d(out, p=0.3, training=self.training)
		out = self.bn4(F.leaky_relu(self.conv5(out)))

		# out = F.dropout2d(out, p=0.3, training=self.training)
		# out = self.bn5(F.leaky_relu(self.conv5(out)))

		# out = F.dropout2d(out, p=0.3, training=self.training)
		# out = self.bn6(F.leaky_relu(self.conv6(out)))

		out = F.dropout2d(out, p=0.3, training=self.training)
		out = F.sigmoid(self.conv7(out))
		# out = self.bn7(out)

		# out = F.sigmoid(self.conv8(out))
		#print('Conv4: ',  out.shape)

		return out

class Discriminator(nn.Module):

	def __init__(self, dim, in_channels):

		super(Discriminator, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels, 512, 3, padding=1,stride=2)
		self.conv2 = nn.Conv2d(512, 512, 3, padding=1)
		self.conv3 = nn.Conv2d(512, 256, 3, padding=1, stride=2)
		self.conv4 = nn.Conv2d(256, 64, 3, padding=1, stride=2)
		
		self.dropout1 = nn.Dropout(p=0.3)
		self.dropout2 = nn.Dropout(p=0.2) 

		self.linear1 = nn.Linear(64 * int(dim/8) * int(dim/8), 200)
		# self.linear2 = nn.Linear(100, 50)
		self.linear3 = nn.Linear(200, 1)

		self.bn1 = nn.BatchNorm2d(512)
		self.bn2 = nn.BatchNorm2d(512)
		self.bn3 = nn.BatchNorm2d(256)
		self.bn4 = nn.BatchNorm2d(64) 

		for m in self.modules():
			if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
				print('Initializing', m)
				nn.init.xavier_normal_(m.weight)

	def forward(self, x):

		# print('Discriminator')

		out = self.bn1(F.leaky_relu(self.conv1(x)))
		# print(out.shape)
		out = self.bn2(F.leaky_relu(self.conv2(out)))
		# print(out.shape)
		out = self.bn3(F.leaky_relu(self.conv3(out)))
		# print(out.shape)
		out = self.bn4(F.leaky_relu(self.conv4(out)))
		# print('conv4', out.shape)
		# print(x.shape[0])
		out = out.view(x.shape[0], -1)
		# print('reshaped ', out.shape)
		out = F.leaky_relu(self.linear1(out))

		out = self.dropout1(out)
		# out = F.relu(self.linear2(out))
		# out = self.dropout2(out)
		out = F.leaky_relu(self.linear3(out)) # uncomment for wgAn
		# out = F.sigmoid(self.linear3(out))
		return out



class AutoEncoder(nn.Module):
	"""
		Autoencoder
	"""

	def __init__(self, out_channels=1, train=True):
		super(AutoEncoder, self).__init__()
		self.encoder = Encoder()
		self.decoder = ColorDecoderConvTrans(out_channels=out_channels)
		# self.decoder = ColorDecoder()




	def forward(self, x):
		out = self.encoder(x)
		out = self.decoder(out)

		return out

class Edge(nn.Module):
	def __init__(self, cuda=True):
		super(Edge, self).__init__()
		self.x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
		self.y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
		self.convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
		self.convy = nn.Conv2d(1, 1, kernel_size=3 , stride=1, padding=1, bias=False)
		self.weights_x = torch.from_numpy(self.x_filter).float().unsqueeze(0).unsqueeze(0)
		self.weights_y = torch.from_numpy(self.y_filter).float().unsqueeze(0).unsqueeze(0)
		self.cuda = cuda
		if self.cuda:
			self.weights_x = self.weights_x.cuda()
			self.weights_y = self.weights_y.cuda()

		self.convx.weight = nn.Parameter(self.weights_x)
		self.convy.weight = nn.Parameter(self.weights_y)

		
		self.convx.weight.requires_grad = False
		self.convy.weight.requires_grad = False

	def forward(self, x):
		g_x = self.convx(x)
		g_y = self.convy(x)
		self.g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
		print(self.g.requires_grad)

		return self.g

class EdgeLoss(nn.Module):

	def __init__(self, device):
		super(EdgeLoss, self).__init__()
		x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
		y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
		self.weights_x = torch.from_numpy(x_filter).float().unsqueeze(0).unsqueeze(0)
		self.weights_y = torch.from_numpy(y_filter).float().unsqueeze(0).unsqueeze(0)

		
		self.weights_x = self.weights_x.to(device)
		self.weights_y = self.weights_y.to(device)

	
	def forward(self, out, target):


		g1_x = nn.functional.conv2d(out, self.weights_x, padding=1)
		g2_x = nn.functional.conv2d(target, self.weights_x, padding=1)
		g1_y = nn.functional.conv2d(out, self.weights_y, padding=1)
		g2_y = nn.functional.conv2d(target, self.weights_y, padding=1)
		


		g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
		g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))

		return torch.mean((g_1 - g_2).pow(2)), g_1, g_2

class EdgeLossSobel3Channel(nn.Module):

    def __init__(self, device):
        super(EdgeLossSobel3Channel, self).__init__()
        x_filter = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],[[1, 0, -1], [2, 0, -2], [1, 0, -1]],[[1, 0, -1], [2, 0, -2], [1, 0, -1]]])
        y_filter = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]])
        self.weights_x = torch.from_numpy(x_filter).float().unsqueeze(0)
        self.weights_y = torch.from_numpy(y_filter).float().unsqueeze(0)

        
        self.weights_x = self.weights_x.to(device)
        self.weights_y = self.weights_y.to(device)

    
    def forward(self, out, target):


        g1_x = nn.functional.conv2d(out, self.weights_x, padding=1)
        g2_x = nn.functional.conv2d(target, self.weights_x, padding=1)
        g1_y = nn.functional.conv2d(out, self.weights_y, padding=1)
        g2_y = nn.functional.conv2d(target, self.weights_y, padding=1)
        
        # print(g1_x.shape, g1_y.shape)
#         print()


        g_1 = (torch.abs(g1_x) + torch.abs(g1_y))
        g_2 = (torch.abs(g2_x) + torch.abs(g2_y))
        
        
        return torch.mean((g_1 - g_2).pow(2)), g_1, g_2
class EdgeLossLaplace(nn.Module):

	def __init__(self, device):
		super(EdgeLossLaplace, self).__init__()
		lap_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
		self.weights_l = torch.from_numpy(lap_filter).float().unsqueeze(0).unsqueeze(0)

		
		self.weights_l = self.weights_l.to(device)

	
	def forward(self, out, target):


		g1_x = nn.functional.conv2d(out, self.weights_l, padding=1)
		g2_x = nn.functional.conv2d(target, self.weights_l, padding=1)
		# g1_y = nn.functional.conv2d(out, self.weights_y, padding=1)
		# g2_y = nn.functional.conv2d(target, self.weights_y, padding=1)
		


		# g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
		# g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))

		return torch.mean((g_1 - g_2).pow(2)), g_1, g_2

class EdgeLossLaplace3CHANNEL(nn.Module):

	def __init__(self, device):
		super(EdgeLossLaplace3CHANNEL, self).__init__()
		lap_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
		# print('lap1', lap_filter.shape)

		lap_filter = np.array([[lap_filter, lap_filter, lap_filter]])
		# print('lap2', lap_filter.shape)
		self.weights_l = torch.from_numpy(lap_filter).float()

		
		self.weights_l = self.weights_l.to(device)

	
	def forward(self, out, target):

		# print(self.weights_l.shape)

		g_1 = nn.functional.conv2d(out, self.weights_l, padding=1)
		g_2 = nn.functional.conv2d(target, self.weights_l, padding=1)

		return torch.mean((g_1 - g_2).pow(2)), g_1, g_2



def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_net():
	CUDA = torch.cuda.is_available()
	autoencoder = AutoEncoder(out_channels=3)
	discriminator = Discriminator(128, 3)
	print('Autoencoder Params', count_parameters(autoencoder))
	print('Discriminator Params', count_parameters(discriminator))
	tensor = torch.randn(32, 1, 128, 128)
	
	if CUDA :
		autoencoder = autoencoder.to(torch.device("cuda"))
		tensor = tensor.to(torch.device("cuda"))
		discriminator = discriminator.to(torch.device("cuda"))
		val = input()
	 
	out_l = autoencoder(tensor)
	out_disc = discriminator(out_l)
	
	print(out_l.shape)
	print(out_disc.shape)

def test_unet():
	unet_save_file = '4_10000_color_model.pth'
	gen = UNetEncDec(unet_save_file)
	print('Gen  params:', count_parameters(gen))
	disc = Discriminator(128, 3)
	tensor = torch.randn(1,1,128,128)
	out = gen(tensor)
	print(out.shape)
	disc_out = disc(out)
	print('disc out', disc_out.shape)
if __name__ == '__main__':
	# test_net()
	test_unet()