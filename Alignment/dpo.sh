CONFIG=$1
PORT=$(( $RANDOM % 1000 + 32768 ))
# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# force crashing on nccl issues like hanging broadcast
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GRUB_CMDLINE_LINUX_DEFAULT="iommu=soft"
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVLL
export DS_SKIP_CUDA_CHECK=1

accelerate launch --main_process_port $PORT --config_file="deepspeed_zero3_dpo.yaml" --num_processes 1 dpo.py --training_config $CONFIG > ./print_log
