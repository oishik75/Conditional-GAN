# Train on MNIST dataset
python train.py --dataset_name MNIST --image_channels 1 --use_gradient_penalty --logs_dir logs --save_dir models --optimizer Adam --lr 1e-4 --n_epochs 5