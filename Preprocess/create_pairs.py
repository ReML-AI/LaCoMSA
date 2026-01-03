import argparse
import json

parser = argparse.ArgumentParser(
    description="Process and extract multilingual English data."
)
parser.add_argument(
    "--input_file", type=str, required=True, help="Path to the input JSON file."
)
parser.add_argument(
    "--output_file", type=str, required=True, help="Path to the output JSON file."
)
args = parser.parse_args()
file = args.file

en_question2en_answer = {}

with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data[0])
result = []
index = 0
add_count = 0
sampled_data = data
for item in sampled_data:
    if "prompt" in item:
        item["instruction"] = item["prompt"]
    else:
        item["prompt"] = item["instruction"]
    lang = "English" if "en_question" not in item else "LRL"
    if "en_question" not in item and "en_instruction" in item:
        item["en_question"] = item["en_instruction"]
    if lang == "English":
        if "en_question" not in item:
            item["en_question"] = item["instruction"]
        temp = en_question2en_answer.get(item["en_question"], [])
        answers = item["answers"]
        temp += [_["generated"] for _ in answers]
        temp = list(set(temp))
        en_question2en_answer[item["en_question"]] = temp

        add_count += 1

print(add_count)

avg_en = 0
count = 0

for item in sampled_data:
    if "prompt" in item:
        item["instruction"] = item["prompt"]
    else:
        item["prompt"] = item["instruction"]
    lang = "English" if "en_question" not in item else "LRL"
    if lang != "English":
        en_solution = en_question2en_answer.get(item["en_question"], [])
        if len(en_solution) == 0:
            print("hit empty")
        item["en_collected_answer"] = en_solution
        avg_en += len(en_solution)
        count += 1

print(avg_en / count)
print(len(sampled_data))


file = file.replace(".json", "en_collect.json")
print(f"Saving to {file}")
with open(file, "w", encoding="utf-8") as fw:
    json.dump(sampled_data, fw, indent=2, ensure_ascii=False)
