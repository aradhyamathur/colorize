import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):

	def __init__(self):

		super(Encoder, self).__init__()
		self.conv1 = nn.Conv2d(1, 1024, 3, padding=1, stride=2)
		self.conv2 = nn.Conv2d(1024, 512, 3, padding=1)
		self.conv3 = nn.Conv2d(512, 256, 3, padding=1)
		self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
		self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)  
		self.conv6 = nn.Conv2d(1024, 1024, 3, padding=1, stride=2)
		self.conv7 = nn.Conv2d(1024, 2048, 3, padding=1, stride=2)
		self.bn1 = nn.BatchNorm2d(1024)
		self.bn2 = nn.BatchNorm2d(512)
		self.bn3 = nn.BatchNorm2d(256)
		self.bn4 = nn.BatchNorm2d(512)
		self.bn5 = nn.BatchNorm2d(1024)
		self.bn6 = nn.BatchNorm2d(1024)
		self.bn7 = nn.BatchNorm2d(2048)



		for m in self.modules():
			if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
				print('Initializing', m)
				nn.init.xavier_normal_(m.weight)
	
	def forward(self, x):

		#print('ENCODER')

		out = self.bn1(F.relu(self.conv1(x)))

		#print('Conv1:', out.shape)

		out = self.bn2(F.relu(self.conv2(out)))

		#print('Conv2: ', out.shape)

		out = self.bn3(F.relu(self.conv3(out)))

		#print('Conv3: ', out.shape)

		out = self.bn4(F.relu(self.conv4(out)))

		#print('Conv4: ', out.shape)
		out = self.bn5(F.relu(self.conv5(out)))

		out = self.bn6(F.relu(self.conv6(out)))
		
		out = self.bn7(F.relu(self.conv7(out)))
		
		return out


class Decoder(nn.Module):

	def __init__(self, out_channels=1):

		super(Decoder, self).__init__()
		self.upsample1 = nn.Upsample(scale_factor=4)
		self.upsample2 = nn.Upsample(scale_factor=2)

		self.conv1 = nn.Conv2d(2048, 512, 3, padding=1)
		self.conv2 = nn.Conv2d(512, 512, 3, padding=1)
		self.conv3 = nn.Conv2d(512, 1024, 3, padding=1)
		self.conv4 = nn.Conv2d(1024, out_channels, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(512)
		self.bn2 = nn.BatchNorm2d(512)
		self.bn3 = nn.BatchNorm2d(1024)

		for m in self.modules():
			if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
				print('Initializing', m)
				nn.init.xavier_normal_(m.weight)

	def forward(self, x):

		#print('DECODER')

		out = self.bn1(F.relu(self.conv1(x)))
		#print('Conv1 : ', out.shape)

		out = self.upsample1(out)

		out = self.bn2(F.relu(self.conv2(out)))
		#print('Conv2: ', out.shape)

		out = self.upsample2(out)

		out = self.bn3(F.relu(self.conv3(out)))
		#print('Conv3: ', out.shape)

		out = F.relu(self.conv4(out))


		#print('Conv4: ',  out.shape)

		return out


class ColorDecoder(nn.Module):

	def __init__(self, out_channels=1):

		super(ColorDecoder, self).__init__()
		self.upsample1 = nn.Upsample(scale_factor=4)
		self.upsample2 = nn.Upsample(scale_factor=2)

		self.conv1 = nn.Conv2d(2048, 256, 3, padding=1)
		self.conv2 = nn.Conv2d(256, 512, 3, padding=1)
		self.conv3 = nn.Conv2d(512, 1024, 3, padding=1)
		self.conv4 = nn.Conv2d(1024, 512, 3, padding=1)
		self.conv5 = nn.Conv2d(512, 256, 3, padding=1)
		self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
		self.conv7 = nn.Conv2d(128, out_channels, 3, padding=1)

		self.bn1 = nn.BatchNorm2d(256)
		self.bn2 = nn.BatchNorm2d(512)
		self.bn3 = nn.BatchNorm2d(1024)
		self.bn4 = nn.BatchNorm2d(512)
		self.bn5 = nn.BatchNorm2d(256)
		self.bn6 = nn.BatchNorm2d(128)


		for m in self.modules():
			if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
				print('Initializing', m)
				nn.init.xavier_normal_(m.weight)

	def forward(self, x):

		#print('DECODER')
		out = self.bn1(F.relu(self.conv1(x)))
		#print('Conv1 : ', out.shape)

		out = self.upsample1(out)

		out = self.bn2(F.relu(self.conv2(out)))
		#print('Conv2: ', out.shape)

		out = self.upsample2(out)

		out = self.bn3(F.relu(self.conv3(out)))
		#print('Conv3: ', out.shape)

		out = self.bn4(F.relu(self.conv4(out)))

		out = self.bn5(F.relu(self.conv5(out)))

		out = self.bn6(F.relu(self.conv6(out)))

		out = F.relu(self.conv7(out))
		#print('Conv4: ',  out.shape)

		return out


class Generator(nn.Module):

	def __init__(self, train=True):

		super(Generator, self).__init__()
		self.encode = Encoder()

		self.decode_color = ColorDecoder(2)
		# self.decode_color = self.decode_color.cuda("cuda:1")
		self.train_stat = train
		if self.train_stat:
			self.decode1 = Decoder()


	def forward(self, x):

		# print('Generator')

		out = self.encode(x)
		# out = out.cuda("cuda:1")
		out_ab = self.decode_color(out)
		# out_ab = out_ab.cuda("cuda:0")
		if self.train_stat:
			out_l = self.decode1(out)
		
			return out_l, out_ab

		return out_ab

class Discriminator(nn.Module):

	def __init__(self, dim):

		super(Discriminator, self).__init__()
		
		self.conv1 = nn.Conv2d(2, 1024, 3, padding=1,stride=2)
		self.conv2 = nn.Conv2d(1024, 512, 3, padding=1)
		self.conv3 = nn.Conv2d(512, 256, 3, padding=1, stride=2)
		self.conv4 = nn.Conv2d(256, 128, 3, padding=1, stride=2)
		
		self.dropout1 = nn.Dropout(p=0.3)
		self.dropout2 = nn.Dropout(p=0.2) 

		self.linear1 = nn.Linear(128 * int(dim/8) * int(dim/8), 100)
		self.linear2 = nn.Linear(100, 50)
		self.linear3 = nn.Linear(50, 1)

		self.bn1 = nn.BatchNorm2d(1024)
		self.bn2 = nn.BatchNorm2d(512)
		self.bn3 = nn.BatchNorm2d(256)
		self.bn4 = nn.BatchNorm2d(128) 

		for m in self.modules():
			if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
				print('Initializing', m)
				nn.init.xavier_normal_(m.weight)

	def forward(self, x):

		# print('Discriminator')

		out = self.bn1(F.relu(self.conv1(x)))
		# print(out.shape)
		out = self.bn2(F.relu(self.conv2(out)))
		# print(out.shape)
		out = self.bn3(F.relu(self.conv3(out)))
		# print(out.shape)
		out = self.bn4(F.relu(self.conv4(out)))
		# print('conv4', out.shape)
		# print(x.shape[0])
		out = out.view(x.shape[0], -1)
		# print('reshaped ', out.shape)
		out = F.relu(self.linear1(out))

		out = self.dropout1(out)
		out = F.relu(self.linear2(out))
		out = self.dropout2(out)
		out = F.sigmoid(self.linear3(out))
		return out

class ColorDecoderConvTrans(nn.Module):

    def __init__(self, out_channels=1):

        super(ColorDecoderConvTrans, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=4)
        self.upsample2 = nn.Upsample(scale_factor=2)

        self.conv1 = nn.ConvTranspose2d(2048, 1024, 3, padding=1, stride=2, output_padding=1)
        self.conv2 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv3 = nn.ConvTranspose2d(512, 256, 3, padding=1, stride=2, output_padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5 = nn.ConvTranspose2d(512, 256, 3, padding=1, stride=2, output_padding=1)
        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, out_channels, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(128)


        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
                print('Initializing', m)
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        # print(x.shape)
        #print('DECODER')
        out = self.bn1(F.relu(self.conv1(x)))
        #print('Conv1 : ', out.shape)

        # out = self.upsample1(out)

        out = self.bn2(F.relu(self.conv2(out)))
        #print('Conv2: ', out.shape)

        # out = self.upsample2(out)

        out = self.bn3(F.relu(self.conv3(out)))
        #print('Conv3: ', out.shape)

        out = self.bn4(F.relu(self.conv4(out)))

        out = self.bn5(F.relu(self.conv5(out)))

        out = self.bn6(F.relu(self.conv6(out)))

        out = F.relu(self.conv7(out))
        #print('Conv4: ',  out.shape)

        return out


class AutoEncoder(nn.Module):
    """
        Autoencoder
    """

    def __init__(self, train=True):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = ColorDecoderConvTrans(out_channels=1)

    def forward(self, x):
        out = self.encoder(x)
        # print(out.shape)
        out = self.decoder(out)
        # print(out.shape)
        return out

class ColorEncoder(nn.Module):
    """
        Autoencoder
    """

    def __init__(self, train=True):
        super(ColorEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = ColorDecoderConvTrans(out_channels=2)

    def forward(self, x):
        out = self.encoder(x)
        # print(out.shape)
        out = self.decoder(out)
        # print(out.shape)
        return out



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_net():
	CUDA = torch.cuda.is_available()
	colorencoder = 	ColorEncoder()

	print('Autoencoder Params', count_parameters(colorencoder))
	tensor = torch.randn(5, 1, 128, 128)
	
	if CUDA :
		colorencoder = colorencoder.cuda()
		tensor = tensor.cuda()
		# val = input()
	 
	out_l = colorencoder(tensor)
	
	print(out_l.shape)

if __name__ == '__main__':
	test_net()