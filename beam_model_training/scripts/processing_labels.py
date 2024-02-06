from pathlib import Path
import json

# Define the directory path
dir_path = Path("F:/labeling_outputs/dora")

# List all the image file names in the directory
image_files = [
    f.name
    for f in dir_path.glob("*")
    if f.suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]
]

# Find the JSON file in the directory
json_file_name = next(dir_path.glob("*.json")).name

# Open and read the JSON file
with open(dir_path / json_file_name, "r") as json_file:
    data = json.load(json_file)

# Extract the required field from each element in the list
json_list = [item["data"]["image"] for item in data]

# Initialize an empty dictionary to store the results
results = {}

# Check whether there is an element in the first list that ends with the name of one of the image files
for json_element in json_list:
    for image_file in image_files:
        if json_element.endswith(image_file):
            # If a match is found, store the result in the dictionary
            results[image_file] = "Found"

# For any image file not found in the JSON list, mark it as 'Not Found'
for image_file in image_files:
    if image_file not in results:
        results[image_file] = "Not Found"

for file_name, status in sorted(results.items()):
    print(f"| {file_name} | {status} |")
