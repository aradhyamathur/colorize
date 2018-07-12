import torch
import torch.nn as nn
from skimage import io
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
from model import Edge
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
def edge1():

	conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
	edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
	conv.weight = nn.Parameter(torch.from_numpy(edge_kernel).float().unsqueeze(0).unsqueeze(0))

	image = io.imread('../datasets/128dim_slices/slices/color_slices/0.png')
	image = color.rgb2gray(image)
	print(image.max(), image.min())
	plt.imshow(image, cmap=plt.cm.gray); plt.show()

	img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
	print(img_tensor.requires_grad)
	out = conv(img_tensor)
	print(out.requires_grad)
	out_img = out.squeeze(0).squeeze(0).detach().numpy()
	plt.imshow(out_img, cmap=plt.cm.gray); plt.show()

def edge2():
	
	edge = Edge()
	
	
	image = io.imread('../datasets/128dim_slices/slices/color_slices/0.png')
	image = color.rgb2gray(image)
	print(image.max(), image.min())
	plt.imshow(image, cmap=plt.cm.gray); plt.show()

	img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

	print(img_tensor.requires_grad)
	out = edge(img_tensor)

	loss = F.mse_loss(out, img_tensor)
	loss.backward()
	print(edge.g.grad)
	print(out.requires_grad)
	out_img = out.squeeze(0).squeeze(0).detach().numpy()
	plt.imshow(out_img, cmap=plt.cm.gray); plt.show()

if __name__ == '__main__':
	edge1()