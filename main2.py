# import matplotlib
# matplotlib.use('Agg')
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
import torch.nn.functional as F
from tensorboard_logger import configure, log_value
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import *






parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='path to data folder', required=True)
parser.add_argument('--image_dim', type=int, help='image dimensions', required=True)
parser.add_argument('--load_prev_model_gen', help='path to previous model')
parser.add_argument('--load_prev_model_disc', help='decoder path')
parser.add_argument('--batch_size_train', type=int, help="train batch size")
parser.add_argument('--batch_size_test', type=int, help="test batch size")
# parser.add_argument('--load_prev_model_disc', help='discriminator path')
parser.add_argument('--reset_files', help='reset file stats(True/False)')
parser.add_argument("--start_epoch", type=int, help="specify start epoch to continue from")
parser.add_argument("--end_epoch", type=int, help="specify end epoch to continue to")
parser.add_argument("--learning_rate_ae", type=float ,help="learning rate")
parser.add_argument("--learning_rate_color", type=float ,help="learning rate")
parser.add_argument("--test_mode", type=bool, help="run in test mode") 

args = parser.parse_args()

if args.test_mode:
	print('.....................................RUNNING IN TEST MODE................................')


LOG_DIR = './log_dir/'

np.set_printoptions(threshold=np.nan)

summary_writer = None 

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVED_MODEL_DIR = './trained_models/'
RANDOM_OUTPUTS_DIR = './rand_outputs/'
EVAL_DIR = './test_outputs/'

if not os.path.exists(SAVED_MODEL_DIR):
	os.makedirs(SAVED_MODEL_DIR)
if not os.path.exists(RANDOM_OUTPUTS_DIR):
	os.makedirs(RANDOM_OUTPUTS_DIR)

if not os.path.exists(LOG_DIR):
	os.makedirs(LOG_DIR)



def train(g_model, d_model, learning_rate_ae, learning_rate_color, train_dataloader, test_dataloader, now):
	
	print("Total Train batches :", len(train_dataloader), "Total test batches:", len(test_dataloader))
	global summary_writer
	draw_iter = 10
	all_save_iter = 500
	cur_save_iter = 100
	test_iter = 100
	if args.test_mode:
		draw_iter = 1
		all_save_iter = 1
		cur_save_iter = 1
		test_iter = 1

	cur_model_dir = SAVED_MODEL_DIR + now + '/'
	filenames = os.listdir(args.data_path + COLOR_DIR)

	l_criteron = nn.MSELoss()
	ab_criterion = nn.BCELoss()

	if args.start_epoch:
		start_epoch = args.start_epoch
	else:start_epoch = 0

	if args.end_epoch:
		end_epoch = args.end_epoch
	else:end_epoch = 100

	criterion_ae = nn.MSELoss()
	optimizer_g = optim.Adam(g_model.parameters(), lr=learning_rate_ae)
	optimizer_d = optim.Adam(d_model.parameters(), lr=learning_rate_color)

	save_model_info(g_model, d_model, cur_model_dir, start_epoch, end_epoch, learning_rate_ae, learning_rate_color, optimizer_g, optimizer_d)

	d_prev = 0.0
	d_iter = 1
	for i in range(start_epoch, end_epoch):
		for j, (x, (y_l, y_ab)) in enumerate(train_dataloader):

			g_model.train()
			d_model.train()
			
			target_y = torch.ones(len(y_ab)).to(device)
			target_x = torch.zeros(len(y_ab)).to(device)
			
			g_model.train_stat = True
			correct = 0
			d_iter = 1
			
			if d_prev > 10.0:
				d_iter = 3
			elif d_prev > 20.0:
				d_iter = 5
			else:
				d_iter = 1
			
			x = x + torch.randn(x.shape)
			x = x.to(device)
			y_l = y_l.to(device)
			y_ab = y_ab.to(device)

			optimizer_g.zero_grad()
			optimizer_d.zero_grad()

			for k in range(d_iter):
				_, out_ab = g_model(x)
				d_real = d_model(y_ab)
				d_fake = d_model(out_ab.detach())
				d_loss_real = ab_criterion(d_real, target_y)
				d_loss_fake = ab_criterion(d_fake, target_x)
				d_loss = d_loss_fake + d_loss_real
				d_loss.backward()

				optimizer_d.step()
				optimizer_d.zero_grad()
				optimizer_g.zero_grad()

			d_prev = d_loss.item()

			out_l, out_ab = g_model(x)
			d_fake = d_model(out_ab)
			loss_l = 1e-4 * criterion_ae(out_l, y_l)
			g_loss = ab_criterion(d_fake, target_y)
			loss_gen =  5.0 * g_loss + loss_l

			loss_gen.backward()
			optimizer_g.step()

#####################################################################################################
			# target_y = torch.ones(len(y_ab)).to(device)
			# target_x = torch.zeros(len(y_ab)).to(device)
			
			# disc_out_real = d_model(y_ab)
			# correct = torch.sum(torch.round(disc_out_real) == target_y).item()
			# loss_real = criterion_color(disc_out_real, target_y) 
			# loss_real.backward()

			# out_l, out_ab = g_model(x)

			# disc_out_fake = d_model(out_ab.detach())
			# correct += torch.sum(torch.round(disc_out_fake) == target_x).item()
			# loss_fake = criterion_color(disc_out_fake, target_x) 
			# loss_fake.backward()
			# optimizer_d.step()

			# for k in range(1):
			# 	out_l, out_ab = g_model(x)
			# 	disc_out_fake = d_model(out_ab)
			# 	loss_l = criterion_ae(out_l, y_l)
			# 	# loss_ab_gen = 0.5 * torch.mean((torch.log(disc_out_fake) - torch.log(1 - disc_out_fake))**2)
			# 	loss_ab_gen = criterion_color(disc_out_fake, target_y)
			# 	loss_gen  =  0.5 * 	loss_l + loss_ab_gen
			# 	loss_gen.backward()
			# 	optimizer_g.step()
####################################################################################################################
			# value = 'Iter : %d Batch: %d D loss real: %.4f D Loss fake: %.4f AE loss: %.4f GEN Color Loss: %.4f\n'%(i, j, loss_real.item(), loss_fake.item(), loss_l.item(), loss_ab_gen.item())

			value = 'Iter : %d Batch: %d D loss  %.4f AE loss: %.4f GEN Color Loss: %.4f G_loss %.4f\n'%(i, j, d_loss.item(), loss_l.item(), loss_gen.item(), g_loss.item())
			print(value)
			summary_writer.add_scalar("D loss", d_loss.item())
			summary_writer.add_scalar("AE loss", loss_l.item())
			summary_writer.add_scalar('GEN  Loss', loss_gen.item())

			update_readings(cur_model_dir + 'train_loss_batch.txt', value)
			if j % draw_iter == 0:
				draw_outputs(i, g_model, now, args.data_path, filenames, j)
			
			if j % all_save_iter == 0:
				print('..SAVING MODEL')
				torch.save(g_model.state_dict(), cur_model_dir + 'colorize2gen_' + str(i) + '.pt')
				print('GEN SAVED')
				print('..SAVING MODEL')
				torch.save(d_model.state_dict(), cur_model_dir + 'colorize2disc_' + str(i) + '.pt' )	
				print('Disc SAVED')

			if j % cur_save_iter == 0:
				print('SAVING MODEL')
				torch.save(g_model.state_dict(), cur_model_dir + 'colorize_gen_cur.pt')
				print('SAVED CURRENT')

			if (j % test_iter == 0 and j != 0) or args.test_mode:
				test_losses = test_model(g_model, test_dataloader, i, now, j)
				avg_test_loss = np.average(test_losses)
				summary_writer.add_scalar("Test loss", avg_test_loss)
				print('Test loss Avg: ', avg_test_loss)
				test_loss_val = '%d, %.4f\n' % (i, avg_test_loss)
				update_readings(cur_model_dir + 'test_loss_avg.txt', test_loss_val)

			if args.test_mode:
				break
		if args.test_mode:
			break

def test_model(model, test_loader, epoch, now, batch_idx, test_len=100):

	global summary_writer

	if not os.path.exists(EVAL_DIR + now):
		os.makedirs(EVAL_DIR + now)
	model.eval()
	model.train_stat = False
	test_losses = []
	diffs_avg = []
	with torch.no_grad():
		for i, (name, x, (y_l,y_ab)) in enumerate(test_loader):

			x = x.to(device)
			y_l = y_l.to(device)
			y_ab = y_ab.to(device)

			inputs = x.cpu()
			# print('Inputs shape ', inputs.shape)
			
			out_ab = model(x)
			loss = F.mse_loss(out_ab, y_ab)
			print('Test batch %d Loss %.4f'%(i, loss.item()))
			test_losses.append(loss.item())
			# print('got output')
			# print('outputs shape', output.shape)
			
			out_ab = out_ab.permute(0, 2, 3, 1)
			output = out_ab.cpu().numpy()
			j = random.randint(0, len(output) - 1)
			a_channel = output[j][:, : , 0]
			b_channel = output[j][:, :, 1]
			image = inputs[j].squeeze(0).numpy()
			# print(image.shape, a_channel.shape, b_channel.shape)

			actual_color_image = plt.imread(args.data_path + COLOR_DIR + name[j])
			true_ab = cv2.cvtColor(actual_color_image, cv2.COLOR_RGB2LAB)[:, :, 1:]


			np_image = np.dstack((image, a_channel, b_channel))
			np_rgb = cv2.cvtColor(np_image, cv2.COLOR_LAB2RGB)
			
			diffs = color_diff(np.dstack((a_channel, b_channel)),  true_ab, image)
			# import ipdb; ipdb.set_trace()


			file_name = EVAL_DIR + now + '/' + 'cimg_' + str(epoch) + '_' + str(batch_idx)+ '_' + str(j) + '_'   + name[j]
			# exit()
			image_scan = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
			grid = make_grid(actual_color_image, np_rgb, image_scan)

			summary_writer.add_image("test image/" + 'cimg_' + str(epoch) +'_'+ str(batch_idx)+ '_' + str(j) + '_' + name[j], grid)
			

			imsave(file_name, np_rgb)
			# cv2.imwrite(file_name.split('.png')[0] + '_mri.png', image)

			diffs_avg.append(np.average(diffs))




			with open(EVAL_DIR+now+'/order.txt', 'a') as f:
				val = "%d, %s\n" % (epoch, 'cimg_' + str(epoch)+ '_' + name[j])
				f.writelines(val)

			if i % 100 == 0 and i !=0:
				break
				if args.test_mode:
					break
			if args.test_mode:
				break
		summary_writer.add_scalar('lab difference', np.average(diffs_avg))
	return test_losses


def draw_outputs(epoch, model, now, dset_path, filenames, batch_idx):

	global summary_writer

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
		for i,  index in enumerate(indices):
			image = cv2.imread(base_dir + filenames[index])
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			
			input_image = torch.from_numpy(image_gray).float()
			input_image = input_image.unsqueeze(0)
			input_image = input_image.unsqueeze(0)
			input_image = input_image.to(device)
			
			output = model(input_image)
			output = output.squeeze(0).permute(1, 2, 0)


			np_image = output.cpu().numpy()
	
			a_channel = np_image[:, :,  0]
			b_channel = np_image[:, :,  1] 

			img_composed = np.dstack((image_gray, a_channel, b_channel))
			img_rgb = cv2.cvtColor(img_composed, cv2.COLOR_LAB2RGB)
			color_img = plt.imread(args.data_path + COLOR_DIR + filenames[index])
			grid = make_grid( color_img, img_rgb, image)
			summary_writer.add_image("random image/" + 'cimg_'+ str(epoch) + '_'+ str(batch_idx)+ '_' + str(i) + '_' + filenames[index], grid)
			file_name = (RANDOM_OUTPUTS_DIR + now + '/' +  'cimg_'+ str(epoch) + '_' + str(batch_idx)+ '_' + str(i) + '_' + filenames[index]).strip()

			# save_image_grid(RANDOM_OUTPUTS_DIR + now)

			imsave(file_name.split('.png')[0] + '_mri.png', image)
			imsave(file_name, img_rgb)

			with open(file_name.split('.png')[0] + '_ab.txt','a') as f:
				f.writelines('a channel\n')
				f.writelines(str(a_channel) + '\n')
				f.writelines('b channel\n')
				f.writelines(str(b_channel) + '\n')

			if args.test_mode:
				break
			# exit()




def main():
	global summary_writer
	
	now = str(datetime.datetime.now()) + '/'
	cur_model_dir = SAVED_MODEL_DIR + now
	os.makedirs(cur_model_dir)

	summary_writer = SummaryWriter(LOG_DIR + now)

	if args.reset_files and (args.reset_files == 'False'):
		reset = False
	else:
		reset = True

	if reset:
		print('Resetting files')
		f = open(cur_model_dir + '/'+ 'train_loss_avg.txt','w')
		f.writelines('')
		f.close()
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
		learning_rate_color = 3e-2

	batch_size_train = 5
	batch_size_test = 5
	if args.batch_size_train:
		batch_size_train = args.batch_size_train
	if args.batch_size_test:
		batch_size_test = args.batch_size_test 

	
	X_train, X_test, y_train, y_test = generate_train_test_split(args.data_path)
	train_dataloader = create_dataloader(args.data_path, X_train, y_train, batch_size_train)
	test_dataloader = create_testdataloader(args.data_path, X_test, y_test, batch_size_test)

	g_model = Generator()
	d_model = Discriminator(args.image_dim)
	
	# g_model = nn.DataParallel(g_model)
	# d_model = nn.DataParallel(d_model)

	if args.load_prev_model_disc:
		d_model.load_state_dict(torch.load(args.load_prev_model_disc))
		print('Discriminator loaded successfully')

	if args.load_prev_model_gen:
		g_model.load_state_dict(torch.load(args.load_prev_model_gen))
		print('Generator loaded successfully')
	# exit()
	g_model = g_model.to(device)
	d_model = d_model.to(device)

	train(g_model, d_model, learning_rate_ae, learning_rate_color, train_dataloader, test_dataloader, now)


if __name__ == '__main__':
	main()