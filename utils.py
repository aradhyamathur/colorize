
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

def make_grid(color_img, colored_img, scan_img):

	scan_img  = scan_img / 255.0
	print(scan_img.dtype)

	img_grid = np.concatenate((color_img, colored_img, scan_img ), axis=1)
	return img_grid

def save_model_info(g_model, d_model, DIR, epoch_start, epoch_end, learning_rate_ae, learning_rate_color, optimizer_ae, optimizer_color):
	f = open(DIR + "model_info.txt", 'a')
	g_model_info = 'Color Generator : \n' + str(g_model) + '\n'
	d_model_info = 'Color Discriminator : \n' + str(d_model) + '\n'
	metrics = 'Epoch start : {} epoch end: {} learning_rate_ae : {} learning_rate_color: {} \n'.format(str(epoch_start), str(epoch_end), str(learning_rate_ae), str(learning_rate_color)) + '\n'
	optimizer_ae_str = "AE optimizer: \n " + str(optimizer_ae.state_dict())  + '\n'
	optimizer_color_str = "Color optimizer: \n " + str(optimizer_color.state_dict())  + '\n'
	f.writelines(g_model_info)
	f.writelines(d_model_info)
	f.writelines(metrics)
	f.writelines(optimizer_ae_str)
	f.writelines(optimizer_color_str)
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