
import numpy as np
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
import cv2
from skimage.io import imsave
import matplotlib.pyplot as plt
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
	val = name[0]

	for n in name[1:]:
		val += "," + n
	if epoch is not None:
		val += ',' + str(epoch)
	val += ',' + str(batchindex) + '\n'
	with open(filename, 'a') as f:
		f.writelines(val)