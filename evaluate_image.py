import torch
from model import AutoEncoder
import argparse
import os
import skimage
from skimage import io
from torchvision import transforms
from skimage import img_as_float

parser = argparse.ArgumentParser()
parser.add_argument('--load_prev_model_gen', help="model path")
parser.add_argument('--input_dir', help="input image directory")

DEMO_DIR = './demo/'

if not os.path.exists(DEMO_DIR):
	os.makedirs(DEMO_DIR)

args = parser.parse_args()

trans = transforms.Compose([transforms.ToTensor()])

# if not args.device:
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# else:
	# device = torch.device(args.device)


gen = AutoEncoder(out_channels=3)
gen.load_state_dict(torch.load(args.load_prev_model_gen))

gen = gen.to(device)
images = os.listdir(args.input_dir)

gen.eval()

for i, img in enumerate(images):
	print(i, img)
	# exit()
	image = io.imread(args.input_dir + img)
	image = img_as_float(skimage.color.rgb2gray(image))
	t_img = trans(torch.from_numpy(image).float().unsqueeze(0).permute(1,2,0).numpy()).unsqueeze(0).to(device)
	# print(t_img.shape)

	# exit()
	with torch.no_grad():
		out = gen(t_img)

	out_img = out.squeeze(0).permute(1,2,0).cpu().numpy()
	io.imsave( DEMO_DIR + img, out_img)
	# break
