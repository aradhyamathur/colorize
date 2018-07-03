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
from model import ColorEncoder
from da_dataloader_efficient import *
import datetime
from itertools import cycle
import random
import gc
import skimage
from skimage.io import imsave
import resource
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import *






parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='path to data folder', required=True)
parser.add_argument('--image_dim', type=int, help='image dimensions', required=True)
parser.add_argument('--load_prev_model_ae', help='path to previous model')
parser.add_argument('--batch_size_train', type=int, help="train batch size")
parser.add_argument('--batch_size_test', type=int, help="test batch size")
# parser.add_argument('--load_prev_model_disc', help='discriminator path')
parser.add_argument('--reset_files', help='reset file stats(True/False)')
parser.add_argument("--start_epoch", type=int, help="specify start epoch to continue from")
parser.add_argument("--end_epoch", type=int, help="specify end epoch to continue to")
parser.add_argument("--learning_rate_ae", type=float ,help="learning rate")
parser.add_argument("--test_mode", type=bool, help="run in test mode") 

args = parser.parse_args()

if args.test_mode:
	print('.....................................RUNNING IN TEST MODE................................')


LOG_DIR = './da_log_dir/'

np.set_printoptions(threshold=np.nan)

summary_writer = None 

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVED_MODEL_DIR = './da_trained_models/'
RANDOM_OUTPUTS_DIR = './da_rand_outputs/'
EVAL_DIR = './da_test_outputs/'

if not os.path.exists(SAVED_MODEL_DIR):
	os.makedirs(SAVED_MODEL_DIR)
if not os.path.exists(RANDOM_OUTPUTS_DIR):
	os.makedirs(RANDOM_OUTPUTS_DIR)

if not os.path.exists(LOG_DIR):
	os.makedirs(LOG_DIR)



def train(model, learning_rate_ae, train_dataloader, test_dataloader, now):
	
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



	if args.start_epoch:
		start_epoch = args.start_epoch
	else:start_epoch = 0

	if args.end_epoch:
		end_epoch = args.end_epoch
	else:end_epoch = 100

	criterion_da = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate_ae)

	save_model_info(model, cur_model_dir, start_epoch, end_epoch, learning_rate_ae, optimizer)

	for i in range(start_epoch, end_epoch):
		for j, (x, (y_l, y_ab)) in enumerate(train_dataloader):

			model.train()
			
			
			x = x + torch.randn(x.shape)
			x = x.to(device)
			y_l = y_l.to(device)
			y_ab = y_ab.to(device)

			optimizer.zero_grad()

			out = model(x)
			loss = criterion_da(out, y_ab)
			loss.backward()
			optimizer.step()

			value = 'Iter : %d Batch: %d AE loss: %.4f \n'%(i, j, loss.item())
			print(value)
			summary_writer.add_scalar("AE loss", loss.item())

			update_readings(cur_model_dir + 'train_loss_batch.txt', value)
			# if j % draw_iter == 0:
			# 	draw_outputs(i, model, now, args.data_path, filenames, j)
			
			if j % all_save_iter == 0:
				print('..SAVING MODEL')
				torch.save(model.state_dict(), cur_model_dir + 'colorize2ae_' + str(i) + '.pt')
				print('AE SAVED')
				print('..SAVED MODEL')

			if j % cur_save_iter == 0:
				print('SAVING MODEL')
				torch.save(model.state_dict(), cur_model_dir + 'colorize_ae_cur.pt')
				print('SAVED CURRENT')

			if args.test_mode or (j % test_iter == 0 and j != 0) :

				test_losses = test_model(model, test_dataloader, i, now, j)
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
		for i, (name, x, (y_l, y_ab)) in enumerate(test_loader):

			x = x.to(device)
			y_ab = y_ab.to(device)

			inputs = x.cpu()

			out = model(x)
			loss = F.mse_loss(out, y_ab)
			print('Test batch %d Loss %.4f'%(i, loss.item()))

			test_losses.append(loss.item())

			out = out.squeeze(1)
			output = out.cpu().permute(0, 2, 3, 1).numpy()

			j = random.randint(0, len(output) - 1)
			image = inputs[j].squeeze(0).numpy()
			a_channel = output[j][:, :, 0]
			b_channel = output[j][:, :, 1]
			
			image_rgb = np.dstack((image, a_channel, b_channel))
			image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_LAB2RGB)
			actual_color_img = plt.imread(args.data_path + COLOR_DIR + name[j])
			

			file_name = EVAL_DIR + now + '/' + 'cimg_' + str(epoch) + '_' + str(batch_idx)+ '_' + str(j) + '_'   + name[j]
			
			
			grid = make_grid(actual_color_img, image_rgb, image)



			# if args.test_mode:   ############### uncomment if using gui
			# 	print("show grid")
			# 	plt.imshow(grid); plt.show()
				# print(grid.shape)

			# grid_rgb = cv2.cvtColor(grid / 255.0, cv2.COLOR_GRAY2RGB)
			summary_writer.add_image("test image/" + 'cimg_' + str(epoch) +'_'+ str(batch_idx)+ '_' + str(j) + '_' + name[j], grid)
			# exit()
			
			# cv2.imwrite(file_name, grid)
			imsave(file_name, grid)

			with open(EVAL_DIR+now+'/order.txt', 'a') as f:
				val = "%d, %s\n" % (epoch, 'cimg_' + str(epoch)+ '_' + name[j])
				f.writelines(val)

			if i % 100 == 0 and i !=0:
				break
				if args.test_mode:
					break
			if args.test_mode:
				break
		# summary_writer.add_scalar('lab difference', np.average(diffs_avg))
	return test_losses


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
	

	batch_size_train = 20
	batch_size_test = 20
	if args.batch_size_train:
		batch_size_train = args.batch_size_train
	if args.batch_size_test:
		batch_size_test = args.batch_size_test 

	
	X_train, X_test, y_train, y_test = generate_train_test_split(args.data_path)
	train_dataloader = create_dataloader(args.data_path, X_train, y_train, batch_size_train)
	test_dataloader = create_testdataloader(args.data_path, X_test, y_test, batch_size_test)

	colorencoder = ColorEncoder()	

	if args.load_prev_model_ae:
		colorencoder.load_state_dict(torch.load(args.load_prev_model_ae))
		print('AE loaded successfully')


	colorencoder = colorencoder.to(device)

	train(colorencoder, learning_rate_ae, train_dataloader, test_dataloader, now)


if __name__ == '__main__':
	main()