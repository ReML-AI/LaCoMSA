import argparse
import json

import torch
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--model_path", type=str, default="princeton-nlp/Llama-3-Base-8B-SFT-DPO"
)
parser.add_argument(
    "--testset",
    type=str,
    default="../Data/ultrafeedback_binarized/subset/random_3000/en.jsonl",
)
parser.add_argument(
    "--save_name", type=str, default="../Data/ultrafeedback_llama3_8b_sft_dpo_en.json"
)
parser.add_argument("--iter", type=int, default=10)
parser.add_argument("--temp", type=float, default=0.9)

args = parser.parse_args()
data_file = args.testset

with open(data_file, "r") as f:
    data = [json.loads(line) for line in f]

if "prompt" in data[0]:
    input_prompt = [i["prompt"] for i in data]
else:
    input_prompt = [i["instruction"] for i in data]

input_prompt = [[{"role": "user", "content": prompt}] for prompt in input_prompt]

sampling_params = SamplingParams(
    n=args.iter,
    temperature=args.temp,
    max_tokens=4096,
    top_p=1.0,
)
llm = LLM(
    model=args.model_path,
    dtype=torch.bfloat16,
    tensor_parallel_size=1,
    max_model_len=16384,
)

generations = llm.chat(input_prompt, sampling_params)
generated_ = []
for output in generations:
    prompt = output.prompt
    generate_text = [o.text for o in output.outputs]
    generated_.append(generate_text)

assert len(data) == len(generated_)
for i, g in zip(data, generated_):
    if "answers" not in i:
        i["answers"] = []
    for _ in g:
        i["answers"].append({"generated": _})

f = open(args.save_name, "w")
json.dump(data, f, indent=2, ensure_ascii=False)
