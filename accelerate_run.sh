# accelerate run
CUDA_VISIBLE_DEVICES="1,2" \
NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
NCCL_SOCKET_IFNAME="eth0" \
OMP_NUM_THREADS="6" \
python -u -m accelerate.commands.launch --config_file configs/huggingface/accelerate.yaml accelerate_main.py  \
--proj_name panMamba --arch panMamba -b 22 --dataset gf2 --warm_up_epochs 0 \
--num_worker 6 -e 2000 --aug_probs 0. 0. --loss l1ssim --val_n_epoch 20 \
--comment "panMamba small config on gf2 dataset" \
--logger_on --log_metric 