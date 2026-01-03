import argparse
import json
import math
import re

# Set up command line arguments
parser = argparse.ArgumentParser(description="Extract DPO data from feedback data.")
parser.add_argument(
    "--target",
    type=str,
    help="Target file name for input and output",
)
parser.add_argument(
    "--reward_column",
    type=str,
    default="reward",
    help="Column name to use for reward values",
)
parser.add_argument(
    "--weight_column",
    type=str,
    default="weight",
    help="Column name to use for weight values",
)
args = parser.parse_args()

# Use the parsed arguments
target = args.target
REWARD_COLUMN = args.reward_column
if args.weight_column:
    WEIGHT_COLUMN = args.weight_column
else:
    WEIGHT_COLUMN = None

f = open("../Data/feedback_data/{}.json".format(target))
data = json.load(f)

dpo_data = []

dpo_samples = []

for i in data:
    lang = "English" if "en_question" not in i else "LRL"
    if lang != "English":

        sorted_output = [
            g
            for g in sorted(
                i["answers"],
                key=lambda x: x[REWARD_COLUMN],
                reverse=True,
            )
        ]

        temp = []
        for j in range(len(sorted_output) - 1):
            for l in range(j + 1, len(sorted_output)):
                sample = {}
                sample["chosen"] = [
                    {"role": "user", "content": i["instruction"]},
                    {
                        "role": "assistant",
                        "content": sorted_output[j]["generated"],
                    },
                ]
                sample["rejected"] = [
                    {"role": "user", "content": i["instruction"]},
                    {
                        "role": "assistant",
                        "content": sorted_output[l]["generated"],
                    },
                ]
                if WEIGHT_COLUMN:
                    weight1 = sorted_output[j].get(WEIGHT_COLUMN, 1.0)
                    weight2 = sorted_output[l].get(WEIGHT_COLUMN, 1.0)
                else:
                    weight1 = 1.0
                    weight2 = 1.0
                sample["score-diff"] = (
                    weight1 * sorted_output[j][REWARD_COLUMN]
                    - weight2 * sorted_output[l][REWARD_COLUMN]
                )
                if (
                    sample["score-diff"] != 0.0
                    and sorted_output[j]["generated"] != sorted_output[l]["generated"]
                ):
                    temp.append(sample)
        dpo_samples.extend(temp)


ratio = 10
train_data = []
dev_data = []
for i in range(len(dpo_samples)):
    if i % ratio == 0:
        dev_data.append(dpo_samples[i])
    else:
        train_data.append(dpo_samples[i])

print(len(train_data))
print(len(dev_data))

train_filename = f"../Data/preference_data/{target}-train.json"
dev_filename = f"../Data/preference_data/{target}-dev.json"

with open(train_filename, "w") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open(dev_filename, "w") as f:
    json.dump(dev_data, f, indent=2, ensure_ascii=False)
