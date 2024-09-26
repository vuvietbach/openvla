#!/bin/bash
#SBATCH --job-name=test    # Specify the job name
#SBATCH --output=slurm_output/test.out           # Output file (%j will append job ID)
#SBATCH --error=slurm_output/test.err            # Error file (%j will append job ID)
#SBATCH --ntasks=1                       # Run on a single CPU
#SBATCH --gpus=1                 # Number of GPUs per node

# Your commands go here:
conda activate ot
cd /home/user01/aiotlab/bachvv/openvla
WANDB_MODE=disable python train.py \
  --vla.type "mamba" \
  --data_root_dir tmp \
  --run_root_dir runs \
  --use_mamba True
