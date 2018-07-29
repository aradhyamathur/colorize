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
	python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --test_mode True --device cuda
	# python main_edge_detect.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --test_mode True --device cpu

run_edge:
	python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --device cuda
	# python main_edge_detect.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --device cuda

test_edge_detect:
	python edge_detect_conv.py