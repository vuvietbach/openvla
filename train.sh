#!/bin/bash
#SBATCH --job-name=pytorch_distributed     # Name of your job
#SBATCH --nodes=3                          # Number of nodes
#SBATCH --nodelist=slurmnode3,slurmnode4,slurmnode5   # Specify exact nodes
#SBATCH --ntasks-per-node=1                # Number of processes per node (one per GPU)
#SBATCH --output=slurm-%j.out              # Output file
#SBATCH --error=slurm-%j.err              # Output file

# conda init
# conda activate openvla

# Set the master address and port for communication (use $SLURM_NODELIST to get node name)
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=36000  # Arbitrary port number for communication

# Set the number of nodes and GPUs
WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

# Run torchrun with the appropriate parameters
torchrun \
    --nproc_per_node=1 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --vla.type "mamba" \
    --data_root_dir tmp \
    --run_root_dir runs \
    --use_mamba True