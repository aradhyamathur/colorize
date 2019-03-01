import torch
from model import AutoEncoder
import argparse
import os
import torch.nn as nn
import skimage
from skimage import io
from torchvision import transforms
from skimage import img_as_float
from shutil import copyfile
from skimage.transform import rescale
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--load_prev_model_gen', help="model path")
parser.add_argument('--input_dir', help="input image directory")
parser.add_argument('--color_dir', help='colr image directory')
parser.add_argument('--custom_name', nargs="?", help='custom name')
parser.add_argument('--image_scale', nargs="?", type=float)
DEMO_DIR = './demo_2/'


if not os.path.exists(DEMO_DIR):
	os.makedirs(DEMO_DIR)

args = parser.parse_args()
model_name = args.load_prev_model_gen.split('/')[-2]

if args.custom_name:
	model_name = model_name + args.custom_name

if not os.path.exists(model_name):
	os.makedirs(DEMO_DIR+model_name)

# print(args.load_prev_model_gen.split('/')[-2])
# exit(0)

trans = transforms.Compose([transforms.ToTensor()])

# if not args.device:
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# else:
	# device = torch.device(args.device)

# gen = AutoEncoder(out_channels=3)
gen = AutoEncoder(out_channels=3)
gen.load_state_dict(torch.load(args.load_prev_model_gen))

gen = gen.to(device)
images = os.listdir(args.input_dir)

gen.eval()

for i, img in enumerate(images):
	# if i % 20 == 0:
	print(i, img.split('.'))
	# exit()
	image = io.imread(args.input_dir + img)
	if args.image_scale:
		image = rescale(image, args.image_scale)
	image = img_as_float(skimage.color.rgb2gray(image))
	# image = np.expand_dims(image, axis=0)
	# image = np.repeat(image, 3, axis=0)
	t_img = trans(torch.from_numpy(image).float().unsqueeze(0).permute(1,2,0).numpy()).unsqueeze(0).repeat(1,3,1,1).to(device)
	# print(t_img.shape)

	# exit()
	with torch.no_grad():
		out = gen(t_img)

	out_img = out.squeeze(0).permute(1,2,0).cpu().numpy()
	io.imsave( DEMO_DIR + model_name+ '/' + img, out_img)
	copyfile(args.color_dir + img, DEMO_DIR + model_name + '/' + img.split('.')[0] + '_color.png')
	copyfile(args.input_dir + img, DEMO_DIR + model_name + '/' + img.split('.')[0] + '_scan.png')
	# break
