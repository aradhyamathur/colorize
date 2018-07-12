
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

def make_grid(gray_img, ae_img, edge_img):

	# print(gray_img.shape, ae_img.shape, edge_img.shape)
	# scan_img  = scan_img / 255.0
	# gray_img = gray_img / 255.0
	# ae_img = ae_img / 255.0
	img_grid = np.concatenate((gray_img, ae_img, edge_img), axis=1)
	return img_grid

def save_model_info(model, DIR, epoch_start, epoch_end, learning_rate_ae, optimizer_ae):
	f = open(DIR + "model_info.txt", 'a')
	model_info = 'AE : \n' + str(model) + '\n'
	metrics = 'Epoch start : {} epoch end: {} learning_rate_ae : {}\n'.format(str(epoch_start), str(epoch_end), str(learning_rate_ae)) + '\n'
	optimizer_ae_str = "AE optimizer: \n " + str(optimizer_ae.state_dict())  + '\n'
	f.writelines(model_info)
	f.writelines(metrics)
	f.writelines(optimizer_ae_str)
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