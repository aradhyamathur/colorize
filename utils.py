
import numpy as np
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
import cv2
from skimage.io import imsave
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
# matplotlib.pyplot.ion()

def update_readings(filename, reading):
	f = open(filename, 'a')
	f.writelines(reading)
	f.close() 

def normalize(image):
	image = (image - image.min()) / (image.max() - image.min())
	return image

# def make_grid(gray_img, ae_img, edge_img_in, edge_img_out):

# 	img_grid = np.concatenate((gray_img, ae_img, edge_img_in, edge_img_out), axis=1)
# 	return img_grid

def save_model_info(model_g, model_d,learning_rate_gen, learning_rate_disc, DIR, epoch_start, epoch_end, learning_rate_ae, optimizer_gen, optimizer_disc, description=None):
	f = open(DIR + "model_info.txt", 'a')
	model_info = 'GEN : \n' + str(model_g) + '\n' + 'Disc :\n' + str(model_d) + '\n'
	metrics = 'Epoch start : {}, epoch end: {}, learning_rate_gen : {}, learning_rate_disc : {}\n'.format(str(epoch_start), str(epoch_end), str(learning_rate_gen),str(learning_rate_disc)) + '\n'
	optimizer_gen_str = "Disc optimizer: \n " + str(optimizer_gen.state_dict())  + '\n'
	optimizer_disc_str = "Disc optimizer: \n " + str(optimizer_disc.state_dict())  + '\n'
	f.writelines(model_info)
	f.writelines(metrics)
	f.writelines(optimizer_gen_str)
	f.writelines(optimizer_disc_str)
	if description is not None:
		f.writelines(description + '\n')
	f.close()


def color_diff(img1, img2, img_scan):
	deltas = []
	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
			color1_lab = LabColor(img_scan[i][j], img1[i][j][0], img1[i][j][1])
			color2_lab = LabColor(img_scan[i][j], img2[i][j][0], img2[i][j][1])
			delta_e = delta_e_cie2000(color1_lab, color2_lab)
			deltas.append(delta_e)

	return deltas

def save_batch_image_names(name, filename, batchindex, epoch=None):
	val = str(name[0])

	for n in name[1:]:
		val += "," + str(n)
	if epoch is not None:
		val += ',' + str(epoch)
	val += ',' + str(batchindex) + '\n'
	with open(filename, 'a') as f:
		f.writelines(val)

def load_external(model, path):
	external_model = torch.load(path)
	mods_weights = []
	for val in external_model.keys():
		if 'encoder' in val and 'bn' not in val and 'weight' in val:
			mods_weights.append(val)
	mods_bias = []
	for val in external_model.keys():
		if 'encoder' in val and 'bn' not in val and 'bias' in val:
			mods_bias.append(val)
	# print(len(mods_bias), len(mods_weights), len(list(model.encoder.children())))
	# print(list(model.encoder.children()))
	for i, val in enumerate(list(model.encoder.children())[:4]):
		val.weight.data = external_model.get(mods_weights[i])
		val.bias.data = external_model.get(mods_bias[i])
	return model



def test_load_ext_model():
	from model import AutoEncoder
	path = '../segmentation vol_generate/data/128dim_slices/fcn/weights.pth'
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = AutoEncoder(3).to(device)
	# model = nn.DataParallel(model)
	model = load_external(model, path)
	model.encoder.requires_grad = False
	model = nn.DataParallel(model)
	tensor = torch.randn(1, 1, 128, 128)
	tensor = tensor.to(device)

	out = model(tensor)
	# print(out.shape)

if __name__ == '__main__':
	test_load_ext_model()

