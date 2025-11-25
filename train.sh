export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1
torchrun \
  --nproc_per_node 1 \
  --master_port 29500 \
  train.py