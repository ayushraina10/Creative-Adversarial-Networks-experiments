#export CUDA_VISIBLE_DEVICES='1' # edit this if you want to limit yourself to GPU
python3 main_progstyle.py --dataset mnist --dataroot ~/Pytorch_experiments/datasets/mnist/ --outf progstyle_chkpts_scaled --cuda --ngpu 2 --lamb 0 --batchSize 128 --lr 0.0004
