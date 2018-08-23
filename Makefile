run:
	python main2.py --data_path ../datasets/128dim_slices/slices/ --image_dim 128
test:
	python main2.py --data_path ../datasets/128dim_slices/slices/ --image_dim 128 --test_mode True
dataloader:
	python dataloader_efficient.py ../datasets/128dim_slices/slices/

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
	python main_edge_gan.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --test_mode True --device cuda --custom_name wasserstein_vgg
	# python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --test_mode True --device cuda

run_edge_gan:
	python main_edge_gan.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --device cuda --custom_name wasserstein_vgg
	# python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --device cuda

eval_edge_gan:
	# python evaluate_image.py --input_dir ../datasets/128dim_slices/slices/scan_slices/ --load_prev_model_gen ./edge_gan_trained_models/2018-08-20\ 16:22:48.463273_wasserstein/colorize2gen_0.pt
		python evaluate_image.py --input_dir ../datasets/128dim_slices/slices/scan_slices/ --load_prev_model_gen ./edge_gan_trained_models/2018-08-16\ 11:55:30.590807_wasserstein/colorize_gen_cur.pt
		
test_edge_detect_script:
	python edge_detect_conv.py
