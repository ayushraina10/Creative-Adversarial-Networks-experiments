export CUDA_VISIBLE_DEVICES='0,1,2'
#export CUDA_LAUNCH_BLOCKING=1 # edit this if you want to limit yourself to GPU
python3 main_progstyle_normed.py --dataset wikiart --dataroot ~/Pytorch_experiments/datasets/wikiart/ --outf can_wikiart_chkpt_halflr_100_epochs_progressive_normed --cuda --ngpu 2  --batchSize 512 --niter 100 --lr 0.0003 --n_class 27
#--lamb 1