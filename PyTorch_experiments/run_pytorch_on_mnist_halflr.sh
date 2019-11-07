#export CUDA_VISIBLE_DEVICES='1' # edit this if you want to limit yourself to GPU
python3 main.py --dataset mnist --dataroot ~/Pytorch_experiments/datasets/mnist/ --outf can_mnist_chkpt_halflr --cuda --ngpu 2 --lamb 1 --batchSize 128 --lr 0.0002 
