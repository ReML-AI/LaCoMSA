import argparse
import gc
import json
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

loss_fn = nn.CrossEntropyLoss(reduction="none")
parser = argparse.ArgumentParser(description="sampling argument")
parser.add_argument("--begin_index", type=int)
parser.add_argument("--data_length", type=int)
parser.add_argument("--data_file", type=str)
parser.add_argument(
    "--model_name", type=str, default="princeton-nlp/Llama-3-Base-8B-SFT-DPO"
)
parser.add_argument("--cache_dir", type=str, default="~/.cache", help="cache dir")
parser.add_argument("--layer_index", type=int, default=-1, help="layer index")
parser.add_argument(
    "--batch_size",
    type=int,
    default=10,
    help="Batch size for processing the data.",
)
parser.add_argument("--save_dir", type=str, default="tmp_lacomsa")
parser.add_argument(
    "--alpha",
    type=float,
    default=1 / 2,
    help="Softmax scale for failures-first weight.",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=1.0,
    help="Post-softmax exponent for failures-first weight.",
)

args = parser.parse_args()

BATCH_SIZE = args.batch_size
data_path = "../Data/{}".format(args.data_file)
target_dir = f"../Data/{args.save_dir}_{args.layer_index}/{args.data_file}"
os.makedirs(target_dir, exist_ok=True)
target_path = target_dir + "/{proc}.json".format(proc=args.begin_index)

data = []
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    cache_dir=args.cache_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    cache_dir=args.cache_dir,
)
if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token


def cosine_similarity_scaled(list1: torch.Tensor, list2: torch.Tensor) -> float:
    """
    Normalized cosine similarity for *normalized* embeddings.
    Normalized cosine similarity takes values from [0;1].
    """
    cosine_sim = (
        (
            torch.dot(list1.flatten(), list2.flatten())
            / (torch.norm(list1) * torch.norm(list2))
        )
        .cpu()
        .float()
        .numpy()
    )
    return float((1.0 + cosine_sim) / 2.0)


def compute_hidden_states_without_mask(tokenizer, model, inputs, layer_index=-1):
    """
    Compute the hidden states of the model for the given inputs.
    inputs: list of input strings
    layer_index: index of the layer to extract hidden states from
    """
    inputs = tokenizer(
        inputs, padding=True, return_tensors="pt", truncation=True, max_length=16_384
    ).to("cuda")
    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )

    hidden_state = output.hidden_states[layer_index]

    mean_hidden_state = torch.mul(
        inputs["attention_mask"].unsqueeze(-1).expand(hidden_state.shape),
        hidden_state,
    ).mean(dim=1)

    return mean_hidden_state, None


def compute_hidden_states(tokenizer, model, batch, layer_index=-1, instruct_mask=False):
    """
    Compute the hidden states of the model for the given batch of inputs.
    batch: list of dictionaries with "formatted_data" and "instruction" keys
    layer_index: index of the layer to extract hidden states from
    instruct_mask: whether to apply a mask based on the instruction
    """
    if not instruct_mask:
        return compute_hidden_states_without_mask(
            tokenizer, model, batch, layer_index=layer_index
        )

    eot_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

    inputs = tokenizer(
        [
            tokenizer.apply_chat_template(sample["formatted_data"], tokenize=False)
            for sample in batch
        ],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to("cuda")

    ppl_mask = torch.ones_like(inputs["input_ids"])

    for idx, sample in enumerate(batch):
        inputs_response = tokenizer(
            sample["instruction"],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        )
        length = inputs_response["input_ids"].size(1)
        ppl_mask[idx, :length] = 0
        ppl_mask[idx, inputs["input_ids"][idx] == tokenizer.pad_token_id] = 0
        ppl_mask[idx, inputs["input_ids"][idx] == eot_id] = 0

    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )

    hidden_state = output.hidden_states[layer_index]

    # Multiply hidden states by ppl_mask to zero out tokens that should not contribute,
    # then sum across the sequence dimension and normalize by the number of unmasked tokens.
    mean_hidden_state = torch.mul(
        ppl_mask.unsqueeze(-1).expand(hidden_state.shape),
        hidden_state,
    ).sum(dim=1) / ppl_mask.sum(dim=1).unsqueeze(-1)

    return mean_hidden_state


def multilingual_alignment_reward_function(
    tokenizer,
    model,
    outputs,
    labels_hidden_states=None,
    last_layer_labels_hidden_states=None,
    layer_index=-1,
    instruct_mask=False,
):
    """
    Compute the reward function for multilingual alignment using a pre-trained model.
    outputs: list of generated outputs in the source language.
    labels_hidden_states: pre-computed hidden states of the labels in the target language (e.g., English). If not provided, it will be computed.

    Return: a list of rewards for the output based on the model's predictions.
    """
    prediction_hidden_states = compute_hidden_states(
        tokenizer,
        model,
        outputs,
        layer_index=layer_index,
        instruct_mask=instruct_mask,
    )
    last_layer_prediction_hidden_states = compute_hidden_states(
        tokenizer,
        model,
        outputs,
        layer_index=-1,
        instruct_mask=instruct_mask,
    )

    results = torch.nn.functional.cosine_similarity(
        prediction_hidden_states.unsqueeze(1),
        labels_hidden_states.unsqueeze(0),
        dim=-1,
    ).cpu()  # batch_size, num_reference

    external_lang_last_layer_results = (
        torch.nn.functional.cosine_similarity(
            last_layer_prediction_hidden_states.unsqueeze(1),
            last_layer_labels_hidden_states.mean(dim=0, keepdim=True).unsqueeze(0),
            dim=-1,
        )
        .squeeze()
        .cpu()
    )  # batch_size, num_reference
    internal_lang_last_layer_results = (
        torch.nn.functional.cosine_similarity(
            last_layer_prediction_hidden_states.unsqueeze(1),
            last_layer_prediction_hidden_states.mean(dim=0, keepdim=True).unsqueeze(0),
            dim=-1,
        )
        .squeeze()
        .cpu()
    )
    lang_results = internal_lang_last_layer_results - external_lang_last_layer_results
    gap = -results.mean(dim=-1) + lang_results
    w = torch.sigmoid(args.alpha * gap) ** args.gamma

    gc.collect()
    torch.cuda.empty_cache()
    return results.tolist(), w.tolist(), gap.tolist()


begin_index = args.begin_index
data_length = args.data_length
end_index = min(len(data), begin_index + data_length)
print("begin_index: {}, end_index: {}".format(begin_index, end_index))

result = []
for i in tqdm(range(begin_index, end_index)):
    item = data[i]
    lang = "English" if not item.get("en_answer") else "LRL"
    if lang == "English":
        continue
    if len(item["en_collected_answer"]) == 0:
        continue

    output = [
        {
            "formatted_data": [
                {"role": "user", "content": item["en_question"]},
                {"role": "assistant", "content": it},
            ],
            "instruction": item["en_question"],
        }
        for it in item["en_collected_answer"][:20]
    ]
    english_generated_samples_hidden_states = compute_hidden_states(
        tokenizer, model, output, layer_index=args.layer_index, instruct_mask=True
    )
    last_layer_english_generated_samples_hidden_states = compute_hidden_states(
        tokenizer, model, output, layer_index=-1, instruct_mask=True
    )

    gc.collect()
    torch.cuda.empty_cache()

    lrl_answers = [it["generated"] for it in item["answers"]]
    reward_lists = []
    weight_lists = []
    gap_lists = []

    for i in range(0, len(lrl_answers), BATCH_SIZE):
        batch_input = lrl_answers[i : i + BATCH_SIZE]
        if len(batch_input) == 0:
            continue
        batch_input = [
            {
                "formatted_data": [
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": lrl_answer},
                ],
                "instruction": item["instruction"],
            }
            for lrl_answer in batch_input
        ]

        batch_rewards, batch_weights, batch_gaps = (
            multilingual_alignment_reward_function(
                tokenizer,
                model,
                batch_input,
                english_generated_samples_hidden_states,
                last_layer_english_generated_samples_hidden_states,
                layer_index=args.layer_index,
                instruct_mask=True,
            )
        )
        reward_lists.extend(batch_rewards)
        weight_lists.extend(batch_weights)
        gap_lists.extend(batch_gaps)
        torch.cuda.empty_cache()

    for i, it in enumerate(item["answers"]):
        reward_list = reward_lists[i]
        it["reward-mean"] = sum(reward_list) / len(reward_list)
        it["reward-max"] = max(reward_list)
        it["rewardlist"] = reward_list
        it["gap"] = gap_lists[i]
        it["weight"] = weight_lists[i]

    result.append(item)

with open(target_path, "w", encoding="utf-8") as fw:
    json.dump(result, fw, indent=2, ensure_ascii=False)
print(f"Processed {len(result)} items and saved to {target_path}")
