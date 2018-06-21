import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
import tqdm
import argparse
import os
from model import Generator, Discriminator
from dataloader_efficient import *
import datetime
from itertools import cycle
import random
import gc
import skimage
from skimage.io import imsave
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='path to data folder', required=True)
parser.add_argument('--image_dim', type=int, help='image dimensions', required=True)
parser.add_argument('--load_prev_model_gen', help='path to previous model')
parser.add_argument('--load_prev_model_dec', help='decoder path')
parser.add_argument('--batch_size_train', type=int, help="train batch size")
parser.add_argument('--batch_size_test', type=int, help="test batch size")
parser.add_argument('--load_prev_model_disc', help='discriminator path')
parser.add_argument('--reset_files', help='reset file stats(True/False)')
parser.add_argument("--start_epoch", type=int, help="specify start epoch to continue from")
parser.add_argument("--end_epoch", type=int, help="specify end epoch to continue to")
parser.add_argument("--learning_rate_ae", type=float ,help="learning rate")
parser.add_argument("--learning_rate_color", type=float ,help="learning rate")
parser.add_argument("--test_mode", type=bool, help="run in test mode") 

args = parser.parse_args()


device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVED_MODEL_DIR = './trained_models/'
RANDOM_OUTPUTS_DIR = './rand_outputs/'

if not os.path.exists(SAVED_MODEL_DIR):
	os.makedirs(SAVED_MODEL_DIR)
if not os.path.exists(RANDOM_OUTPUTS_DIR):
	os.makedirs(RANDOM_OUTPUTS_DIR)



def update_readings(filename, reading):
	f = open(filename, 'a')
	f.writelines(reading)
	f.close() 


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


def train(g_model, d_model, learning_rate_ae, learning_rate_color, train_dataloader, test_dataloader, now):
	
	draw_iter = 1
	all_save_iter = 500
	cur_save_iter = 100

	if args.test_mode:
		draw_iter = 1
		all_save_iter = 1
		cur_save_iter = 1

	cur_model_dir = SAVED_MODEL_DIR + now + '/'
	filenames = os.listdir(args.data_path + COLOR_DIR)

	l_criteron = nn.MSELoss()
	ab_criterion = nn.BCELoss()

	if args.start_epoch:
		start_epoch = args.start_epoch
	else:start_epoch = 0

	if args.end_epoch:
		end_epoch = args.end_epoch
	else:end_epoch = 100000

	criterion_ae = nn.MSELoss()
	criterion_color = nn.BCELoss()
	optimizer_ae = optim.Adam(g_model.parameters(), lr=learning_rate_ae)
	optimizer_color = optim.Adam(d_model.parameters(), lr=learning_rate_color)

	save_model_info(g_model, d_model, cur_model_dir, start_epoch, end_epoch, learning_rate_ae, learning_rate_color, optimizer_ae, optimizer_color)

	loader = cycle(train_dataloader)  
	for i in range(start_epoch, end_epoch):

		# gc.collect()

		g_model.train_stat = True
		correct = 0
		x, (y_l, y_ab) = next(loader)
		
		# x = torch.from_numpy(x)
		# y_l = torch.from_numpy(y_l)
		# y_ab = torch.from_numpy(y_ab)
		
		x = x.to(device)
		y_l = y_l.to(device)
		y_ab = y_ab.to(device)

		optimizer_ae.zero_grad()
		optimizer_color.zero_grad()

		target_y = torch.ones(len(y_ab)).to(device)
		target_x = torch.zeros(len(y_ab)).to(device)
		
		disc_out_real = d_model(y_ab)
		correct = torch.sum(torch.round(disc_out_real) == target_y).item()
		loss_real = criterion_color(disc_out_real, target_y) 
		loss_real.backward()

		out_l, out_ab = g_model(x)

		disc_out_fake = d_model(out_ab.detach())
		correct += torch.sum(torch.round(disc_out_fake) == target_x).item()
		loss_fake = criterion_color(disc_out_fake, target_x) 
		loss_fake.backward()
		optimizer_color.step()
		print('Discriminator Accuracy:', (correct*1.0)/ (2 * len(x)))
		for k in range(5):
			out_l, out_ab = g_model(x)
			disc_out_fake = d_model(out_ab)
			loss_l = criterion_ae(out_l, y_l)

			loss_ab_gen = criterion_color(disc_out_fake, target_y)
			loss_gen  =  0.5 * 	loss_l + loss_ab_gen
			loss_gen.backward()
			optimizer_ae.step()

		value = 'Iter : %d D loss real %.4f D Loss fake %.4f AE loss %.4f\n'%(i, loss_real.item(), loss_fake.item(), loss_l.item())
		print(value)
		
		update_readings(cur_model_dir + 'train_loss_batch.txt', value)
		if i % draw_iter == 0:
			draw_outputs(i, g_model, now, args.data_path, filenames)
		
		if i % all_save_iter == 0:
			print('..SAVING MODEL')
			torch.save(g_model.state_dict(), cur_model_dir + 'colorize2gen_' + str(i) + '.pt')
			print('GEN SAVED')
			print('..SAVING MODEL')
			torch.save(d_model.state_dict(), cur_model_dir + 'colorize2disc_' + str(i) + '.pt' )	
			print('Disc SAVED')

		if i % cur_save_iter == 0:
			print('SAVING MODEL')
			torch.save(g_model.state_dict(), cur_model_dir + 'colorize_gen_cur.pt')
			print('SAVED CURRENT')

		if args.test_mode:
			break

# def test_model(model, test_loader):
# 	model.eval()
# 	with torch.no_grad():


def draw_outputs(epoch, model, now, dset_path, filenames):
	if not os.path.exists(RANDOM_OUTPUTS_DIR+now):
		os.makedirs(RANDOM_OUTPUTS_DIR+now)
	file = open(RANDOM_OUTPUTS_DIR + now + '/order.txt', 'a')
	indices = []
	for i in range(5):
		index = random.randint(0, len(filenames))
		indices.append(index)
		file.writelines(str(epoch) + ',' + filenames[index] + '\n')
	file.close()
	model.train_stat = False
	model.eval()
	model.to(device)
	
	with torch.no_grad():
		images = []
		base_dir = dset_path + SCAN_DIR + '/'
		image_names = os.listdir(base_dir)
		for index in indices:
			image = cv2.imread(base_dir + filenames[index], 0)
			input_image = torch.from_numpy(image).float()
			input_image = input_image.unsqueeze(0)
			input_image = input_image.unsqueeze(0)
			input_image = input_image.to(device)
			output = model(input_image)
			# print(output.shape)
			output = output.squeeze(0).permute(1, 2, 0)
			# print(output.shape)

			np_image = output.cpu().numpy()
			# print(np_image.shape)
	
			a_channel = np_image[:, :,  0]
			b_channel = np_image[:, :,  1] 
			
			# print('a', a_channel)
			# print('b', b_channel)

			
			img_composed = np.dstack((image, a_channel, b_channel))
			img_rgb = cv2.cvtColor(img_composed, cv2.COLOR_LAB2RGB)

			file_name = (RANDOM_OUTPUTS_DIR + now + '/' +  'cimg_'+ str(epoch)+ '_' + filenames[index]).strip()

			imsave(file_name.split('.png')[0] + '_mri.png', image)
			imsave(file_name, img_rgb)
			# exit()




def main():
	now = str(datetime.datetime.now()) + '/'
	cur_model_dir = SAVED_MODEL_DIR + now
	os.makedirs(cur_model_dir)

	if args.reset_files and (args.reset_files == 'False'):
		reset = False
	else:
		reset = True

	if reset:
		print('Resetting files')
		# f = open(cur_model_dir + '/'+ 'train_loss_avg.txt','w')
		# f.writelines('')
		# f.close()
		f = open(cur_model_dir + '/'+ 'train_loss_batch.txt', 'w')
		f.writelines('')
		f.close()

		f = open(cur_model_dir + '/'+ 'test_loss_avg.txt','w')
		f.writelines('')
		f.close()
	else:
		print('no reset , appending to former data')



	if args.learning_rate_ae:
		learning_rate_ae = args.learning_rate_ae
	else:
		learning_rate_ae = 4e-3
	
	if args.learning_rate_color:
		learning_rate_color = args.learning_rate_color
	else:
		learning_rate_color = 3e-3

	batch_size_train = 5
	batch_size_test = 5
	if args.batch_size_train:
		batch_size_train = args.batch_size_train
	if args.batch_size_test:
		batch_size_test = args.batch_size_test 

	X_train, X_test, y_train, y_test = generate_train_test_split(args.data_path)
	train_dataloader = create_dataloader(args.data_path, X_train, y_train, batch_size_train)
	test_dataloader = create_dataloader(args.data_path, X_test, y_test, batch_size_test)

	g_model = Generator()
	d_model = Discriminator(args.image_dim)

	g_model = g_model.to(device)
	d_model = d_model.to(device)

	train(g_model, d_model, learning_rate_ae, learning_rate_color, train_dataloader, test_dataloader, now)


if __name__ == '__main__':
	main()