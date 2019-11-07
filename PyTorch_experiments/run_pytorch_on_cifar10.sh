#export CUDA_VISIBLE_DEVICES='1' # edit this if you want to limit yourself to GPU
python3 main.py --dataset cifar10 --dataroot ~/Pytorch_experiments/datasets/cifar10/ --outf cifar10_can_chkpts_test --cuda --ngpu 2 --batchSize 128 --lr 0.0004 --lamb 1 --n_class 10 
