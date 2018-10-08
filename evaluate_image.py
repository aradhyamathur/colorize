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
parser = argparse.ArgumentParser()
parser.add_argument('--load_prev_model_gen', help="model path")
parser.add_argument('--input_dir', help="input image directory")
parser.add_argument('--color_dir', help='colr image directory')
parser.add_argument('--custom_name', nargs="?", help='custom name')
parser.add_argument('--image_scale', nargs="?", type=float)
DEMO_DIR = './cgan_demo2/'


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
gen = nn.DataParallel(AutoEncoder(out_channels=3))
gen.load_state_dict(torch.load(args.load_prev_model_gen))

gen = gen.to(device)
images = os.listdir(args.input_dir)

gen.eval()

for i, img in enumerate(images):
	if i % 20 == 0:
		print(i, img.split('.'))
		# exit()
		image = io.imread(args.input_dir + img)
		color_image = io.imread(args.color_dir + img)
		scan_image = io.imread(args.input_dir+img)
		scan_image = skimage.color.gray2rgb(scan_image)
		if args.image_scale:
			image = rescale(image, args.image_scale)
			scan_image = rescale(scan_image, args.image_scale)
			color_image = rescale(color_image, args.image_scale)
		image = img_as_float(skimage.color.rgb2gray(image))
		img_tensor = torch.from_numpy(image).float().unsqueeze(0)
		random_vec = torch.randn(*img_tensor.shape)
		# print(img_tensor.shape)
		# print(random_vec.shape)
		img_tensor = torch.cat((img_tensor, random_vec), 0)
		t_img = trans(img_tensor.numpy()).permute(1,0,2).unsqueeze(0).to(device)
		# print(t_img.shape)

		# exit()
		with torch.no_grad():
			out = gen(t_img)

		out_img = out.squeeze(0).permute(2,1,0).cpu().numpy()
		io.imsave( DEMO_DIR + model_name+ '/' + img, out_img)
		io.imsave(DEMO_DIR + model_name + '/' + img.split('.')[0] + '_color.png', color_image)
		io.imsave(DEMO_DIR + model_name + '/' + img.split('.')[0] + '_scan.png', scan_image)

		# copyfile(args.color_dir + img, DEMO_DIR + model_name + '/' + img.split('.')[0] + '_color.png')
		# copyfile(args.input_dir + img, DEMO_DIR + model_name + '/' + img.split('.')[0] + '_scan.png')
		# exit()
	# break
