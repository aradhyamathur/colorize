import torch
import numpy as np
import cv2
from skimage.io import imsave
from .model import ColorEncoder
import os

OUTPUT_DIR = './EVAL_OUTPUT/'


if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

def evaluate(model, image, file_name):

	input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
	
	if torch.cuda.is_available():
		model = model.cuda()
		input_tensor = input_tensor.cuda()

	output = model(input_tensor)
	output_cpu = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
	a_channel = output_cpu[:, :, 0]
	b_channel = output_cpu[:, :, 1]

	image_lab = np.dstack((image, a_channel, b_channel))
	
	image_rgb = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)

	filename = OUTPUT_DIR + file_name

	imsave(filename, image_rgb)


def main():
	img_name = sys.argv[1]
	img = cv2.imread(img_name)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	model = ColorEncoder()
	model.load_state_dict(torch.load(sys.argv[2]))   


