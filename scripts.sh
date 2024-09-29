export TORCH_USE_CUDA_DSA=1; \
CUDA_LAUNCH_BLOCKING=1 
python vla-scripts/finetune.py
WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir tmp \
  --dataset_name ucsd_pick_and_place_dataset_converted_externally_to_rlds \
  --run_root_dir checkpoints \
  --adapter_tmp_dir tmp \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project vla \
  --wandb_entity aiotlab \
  --save_steps 10

WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir tmp \
  --dataset_name ucsd_pick_and_place_dataset_converted_externally_to_rlds \
  --run_root_dir checkpoints \
  --adapter_tmp_dir tmp \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project vla \
  --wandb_entity aiotlab \
  --save_steps 10

WANDB_MODE=disabled python finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir tmp \
  --dataset_name ucsd_pick_and_place_dataset_converted_externally_to_rlds \
  --run_root_dir checkpoints \
  --adapter_tmp_dir tmp \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project vla \
  --wandb_entity aiotlab \
  --save_steps 10

WANDB_MODE=disabled python finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir tmp \
  --dataset_name ucsd_pick_and_place_dataset_converted_externally_to_rlds \
  --run_root_dir checkpoints \
  --adapter_tmp_dir tmp \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project vla \
  --wandb_entity aiotlab \
  --save_steps 10

pip install dracus
pip install tqdm
pip install accelerate

srun --nodelist=slurmnode8 --pty bash -i
/home/admin/.cache/huggingface/modules/transformers_modules/openvla/openvla-7b/31f090d05236101ebfc381b61c674dd4746d4ce0/modeling_prismatic.py(291)

python train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir tmp \
  --run_root_dir runs

WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 2 train.py \
  --vla.type "mamba" \
  --data_root_dir data \
  --run_root_dir runs \
  --mamba_backbone_id "mamba-codestral-7b"

WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 2 train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir data \
  --run_root_dir runs 

WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 4 train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir data \
  --run_root_dir runs 

WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 4 train.py \
  --vla.type "mamba" \
  --data_root_dir data \
  --run_root_dir runs \
  --mamba_backbone_id "mamba-codestral-7b"

torchrun --standalone --nnodes 1 --nproc-per-node 1 train.py \
  --vla.type "mamba" \
  --data_root_dir data \
  --run_root_dir runs \
  --mamba_backbone_id "mamba-codestral-7b"

WANDB_MODE=disabled python train.py \
  --vla.type "mamba" \
  --data_root_dir data \
  --run_root_dir runs \
  --mamba_backbone_id "mamba-codestral-7b"

WANDB_MODE=disable python train.py \
  --vla.type "mamba" \
  --data_root_dir data \
  --run_root_dir runs \
  --use_mamba True

HF_TOKEN=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

rm -rf models--nguyenvulebinh--envibert \
  models--nguyenvulebinh--wav2vec2-base-vietnamese-250h \
  models--nguyenvulebinh--wav2vec2-base-vi-vlsp2020 \
  models--nguyenvulebinh--wav2vec2-large-vi \
  models--nguyenvulebinh--wav2vec2-large-vi-vlsp2020 \

      --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \

pip install tfds-nightly
pip install apache-beam
pip install tensorflow
pip install draccus
pip install git+https://github.com/moojink/dlimp_openvla#egg=dlimp
pip install tensorflow_graphics
pip install jsonlines
pip install gsutil
pip install wandb
# Change directory to your base datasets folder
cd data ; wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/ ; mv bridge_dataset bridge_orig

du -h /home/mdxuser/* --max-depth=1 | sort -h

pip install draccu