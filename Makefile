run:
	python main2.py --data_path ../datasets/128dim_slices/slices/ --image_dim 128
test:
	python main2.py --data_path ../datasets/128dim_slices/slices/ --image_dim 128 --test_mode True
dataloader:
	python dataloader_rgb.py ../datasets/128dim_slices/slices/

test_ae:
	python main_autoencoder.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --test_mode True
	#python main_autoencoder.py --data_path ../color_my_pytorch_slices/data/slices --image_dim 128 --test_mode True

run_ae:
	python main_autoencoder.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128
	# python main_autoencoder.py --data_path ../color_my_pytorch_slices/data/slices --image_dim 128

test_edge:
	python main_edge_detect.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --test_mode True --device cuda
	# python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --test_mode True --device cuda

run_edge:
	python main_edge_detect.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --device cuda
	# python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --device cuda

test_edge_gan:
	python main_edge_gan.py --data_path ../datasets/random_axis/subslice_f/  --image_dim 128 --test_mode True --device cuda --custom_name random_axis_model_redux_wasserstein_lite_test_sobel --description _lap4_gfast_weight_clip_3_point_0_edge_1point0_tv_loss_1eminus8 
	#--load_segmentation "../segmentation vol_generate/data/128dim_slices/fcn/weights.pth"
	# python main_edge_gan.py --data_path ../datasets/128dim_slices_augmented/slices/  --image_dim 128 --test_mode True --device cuda --custom_name wasserstein_lite
	# python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --test_mode True --device cuda

run_edge_gan:

	python main_edge_gan.py --data_path ../datasets/random_axis/subslice_f/  --image_dim 128 --device cuda --custom_name random_axis_model_redux_wgan_gp_lite_sobel_l1 --description sobel_edge_without_dataparallel_grad_penalty_10
	#--load_segmentation "../segmentation vol_generate/data/128dim_slices/fcn/weights.pth"
# 	python main_edge_gan.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --device cuda --custom_name wasserstein_lite_lap4_edge_mult_1_gfast_weight_clip_5_point_0
	# python main_edge_gan.py --data_path ../datasets/128dim_slices_augmented/slices/  --image_dim 128 --device cuda --custom_name wasserstein_lite
	# python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --device cuda


test_edge_detect_script:
	python edge_detect_conv.py

evaluate_images:
	python evaluate_image.py  --load_prev_model_gen edge_gan_trained_models/2019-01-08\ 17:11:50.358210_model_redux_wasserstein_lite_sobel/colorize2gen_batch_4_2000.pt --input_dir ../datasets/volumes/Slices/scan/  --color_dir ../datasets/volumes/Slices/color/ --custom_name _model_redux_4_2000_complete_dsets  --image_scale 1.0
	#python evaluate_image.py  --load_prev_model_gen edge_gan_trained_models/2018-11-06\ 11:42:52.682790_wasserstein_lite/colorize2gen_batch_1_0.pt --input_dir ../datasets/volumes/Slices/scan/  --color_dir ../datasets/volumes/Slices/color/ --custom_name _withtv_batch_0_2000_complete_dsets  --image_scale 1.0
