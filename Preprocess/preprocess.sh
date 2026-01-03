#!/bin/bash

# Step 1: Sample responses
for l in en es de fr ru; do TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python vllm_sampling.py --model_path princeton-nlp/Llama-3-Base-8B-SFT-DPO --testset ../Data/ultrafeedback_binarized/subset/random_3000/${l}.jsonl --save_name ../Data/ultrafeedback_llama3_8b_sft_dpo_lacomsa_${l}.json; done
for l in es de fr ru; do python create_pairs.py --input_file ../Data/ultrafeedback_llama3_8b_sft_dpo_lacomsa_${l}.json --output_file ../Data/ultrafeedback_llama3_8b_sft_dpo_lacomsa_en_${l}.json; done

# Step 2: Generate LaCoMSA reward signals
for l in fr ru; do TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python LaCoMSA.py --begin_index 0 --data_length 6000 --data_file ultrafeedback_llama3_8b_sft_dpo_lacomsa_en_${l}.json --model_name princeton-nlp/Llama-3-Base-8B-SFT-DPO --layer_index 15 --save_dir tmp_lacomsa_ultrafeedback_llama3_8b_sft_dpo; done

# Step 3: Merge the generated reward signals and extract DPO data
python json_merged.py --source_dir ../Data/tmp_lacomsa_ultrafeedback_llama3_8b_sft_dpo_15 --target_file ../Data/feedback_data/tmp_lacomsa_ultrafeedback_llama3_8b_sft_dpo_15.json --search_sub_dir;
python extract_dpo_data.py --target tmp_lacomsa_ultrafeedback_llama3_8b_sft_dpo_15 --reward_column reward-mean --weight_column weight;
python filter_MSA_data.py --input_file ../Data/preference_data/lacomsa_ultrafeedback_llama3_8b_sft_dpo_15-train.json;
