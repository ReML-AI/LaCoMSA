import random

random.seed(42)  # for reproducibility

import argparse

import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=64)

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
args = parser.parse_args()

df = pd.read_json(args.input_file)
FILE = args.input_file.split("/")[-1].replace("-train.json", "")


selected_samples = []  # take only 1 sample per question
df["question"] = df["chosen"].parallel_apply(lambda x: x[0]["content"])
df["chosen_content"] = df["chosen"].parallel_apply(lambda x: x[1]["content"])
df["rejected_content"] = df["rejected"].parallel_apply(lambda x: x[1]["content"])

for question in df["question"].unique():
    samples = df[df["question"] == question]
    if len(samples) > 1:
        sample = samples.iloc[random.randint(0, len(samples) - 1)]
        selected_samples.append(sample.to_dict())
    else:
        selected_samples.append(samples.iloc[0].to_dict())

df = pd.DataFrame(selected_samples)

print(f"Total samples: {len(df)}")

df = df[["chosen", "rejected", "score-diff"]]
df.to_json(
    f"../Data/preference_data/{FILE}-train.json",
    force_ascii=False,
    indent=4,
    index=False,
)
