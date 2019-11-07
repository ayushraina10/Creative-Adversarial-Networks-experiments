export CUDA_VISIBLE_DEVICES='0,1,2' # edit this if you want to limit yourself to GPU
python3 main.py --dataset wikiart --dataroot ~/Pytorch_experiments/datasets/wikiart/ --outf can_wikiart_chkpt_baseline_256bs --cuda --ngpu 2 --lamb 1 --batchSize 256 --niter 100 --lr 0.0002 --n_class 27
