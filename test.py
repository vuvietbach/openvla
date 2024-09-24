# import tensorflow_datasets as tfds
# import tqdm

# # optionally replace the DATASET_NAMES below with the list of filtered datasets from the google sheet
# DATASET_NAMES = ['fractal20220817_data', 'kuka', 'bridge', 'taco_play', 'jaco_play', 'berkeley_cable_routing', 'roboturk', 'nyu_door_opening_surprising_effectiveness', 'viola', 'berkeley_autolab_ur5', 'toto', 'language_table', 'columbia_cairlab_pusht_real', 'stanford_kuka_multimodal_dataset_converted_externally_to_rlds', 'nyu_rot_dataset_converted_externally_to_rlds', 'stanford_hydra_dataset_converted_externally_to_rlds', 'austin_buds_dataset_converted_externally_to_rlds', 'nyu_franka_play_dataset_converted_externally_to_rlds', 'maniskill_dataset_converted_externally_to_rlds', 'furniture_bench_dataset_converted_externally_to_rlds', 'cmu_franka_exploration_dataset_converted_externally_to_rlds', 'ucsd_kitchen_dataset_converted_externally_to_rlds', 'ucsd_pick_and_place_dataset_converted_externally_to_rlds', 'austin_sailor_dataset_converted_externally_to_rlds', 'austin_sirius_dataset_converted_externally_to_rlds', 'bc_z', 'usc_cloth_sim_converted_externally_to_rlds', 'utokyo_pr2_opening_fridge_converted_externally_to_rlds', 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds', 'utokyo_saytap_converted_externally_to_rlds', 'utokyo_xarm_pick_and_place_converted_externally_to_rlds', 'utokyo_xarm_bimanual_converted_externally_to_rlds', 'robo_net', 'berkeley_mvp_converted_externally_to_rlds', 'berkeley_rpt_converted_externally_to_rlds', 'kaist_nonprehensile_converted_externally_to_rlds', 'stanford_mask_vit_converted_externally_to_rlds', 'tokyo_u_lsmo_converted_externally_to_rlds', 'dlr_sara_pour_converted_externally_to_rlds', 'dlr_sara_grid_clamp_converted_externally_to_rlds', 'dlr_edan_shared_control_converted_externally_to_rlds', 'asu_table_top_converted_externally_to_rlds', 'stanford_robocook_converted_externally_to_rlds', 'eth_agent_affordances', 'imperialcollege_sawyer_wrist_cam', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 'uiuc_d3field', 'utaustin_mutex', 'berkeley_fanuc_manipulation', 'cmu_food_manipulation', 'cmu_play_fusion', 'cmu_stretch', 'berkeley_gnm_recon', 'berkeley_gnm_cory_hall', 'berkeley_gnm_sac_son']
# DATASET_NAMES = ['ucsd_pick_and_place_dataset_converted_externally_to_rlds']
# DOWNLOAD_DIR = 'tmp'

# print(f"Downloading {len(DATASET_NAMES)} datasets to {DOWNLOAD_DIR}.")
# for dataset_name in tqdm.tqdm(DATASET_NAMES):
#   _ = tfds.load(dataset_name, data_dir=DOWNLOAD_DIR)

from transformers import Mamba2Config, Mamba2ForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

HF_TOKEN = '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'
login(HF_TOKEN)
model_id = 'mistralai/Mamba-Codestral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_id, revision='refs/pr/9', from_slow=True, legacy=False)
model = Mamba2ForCausalLM.from_pretrained(model_id, revision='refs/pr/9')
import pdb; pdb.set_trace()
tokenizer.add_special_tokens({"pad_token": "<PAD>"})
print(tokenizer.pad_token_id)
print(tokenizer.pad_token)
print(model.config)
# tokenizer = model.tokenizer
# input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

# out = model.generate(input_ids, max_new_tokens=10)
# print(tokenizer.batch_decode(out))