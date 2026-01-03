import argparse
import json
import os

parser = argparse.ArgumentParser(description="merging")
parser.add_argument("--source_dir", type=str)
parser.add_argument("--target_file", type=str)
parser.add_argument(
    "--search_sub_dir", action="store_true", help="Search subdirectories for JSON files"
)

args = parser.parse_args()

input_folder = args.source_dir
output_file = args.target_file

merged_data = []

if not args.search_sub_dir:
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r") as json_file:
                try:
                    data = json.load(json_file)
                    merged_data = merged_data + data
                except json.JSONDecodeError as e:
                    print(f"无法解析文件 {filename}: {e}")
else:
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r") as json_file:
                    try:
                        data = json.load(json_file)
                        merged_data = merged_data + data
                    except json.JSONDecodeError as e:
                        print(f"Error {filename}: {e}")


print(len(merged_data))


print(len(merged_data))
with open(output_file, "w") as merged_json_file:
    json.dump(merged_data, merged_json_file, indent=4, ensure_ascii=False)
