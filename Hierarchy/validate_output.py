# Your input dictionary
# Load your hierarchy JSON file
import json
with open('output.json', 'r') as f:
    data_dict = json.load(f)


# Read the wnids.txt file and store the IDs in a set
with open("wnids.txt", "r") as file:
    wnids = set(line.strip() for line in file.readlines())

# Extract the IDs from the input dictionary and store them in a set
dict_ids = set()
for ids in data_dict.values():
    dict_ids.update(ids)

# Find the difference between the wnids and the dictionary IDs
difference = wnids.difference(dict_ids)

# Print the difference
print("Difference between wnids and dictionary IDs:", difference)

# Extract the IDs from the input dictionary and store them in a set
dict_ids = set()
repeated_ids = set()

for ids in data_dict.values():
    for id in ids:
        if id in dict_ids:
            repeated_ids.add(id)
        else:
            dict_ids.add(id)

# Print repeated IDs, if any
if repeated_ids:
    print("Repeated IDs found in the dictionary:", repeated_ids)
else:
    print("No repeated IDs found in the dictionary.")

# Find the difference between the wnids and the dictionary IDs
difference = wnids.difference(dict_ids)

# Print the difference
print("Difference between wnids and dictionary IDs:", difference)





wnids = {'n03770439', 'n02948072', 'n03444034', 'n04099969', 'n04285008', 'n04275548', 'n01984695'}

with open("words.txt", "r") as file:
    id_label_dict = {}
    for line in file:
        wnid, label = line.strip().split("\t")
        id_label_dict[wnid] = label

for wnid in wnids:
    print(f"{wnid}: {id_label_dict.get(wnid, 'Not found')}")












import requests

ids_to_find = {'n03770439', 'n02948072', 'n03444034', 'n04099969', 'n04285008', 'n04275548', 'n01984695'}

# Query the ImageNet API for the labels
for id in ids_to_find:
    response = requests.get(f"http://www.image-net.org/api/text/wordnet.synset.getwords?wnid={id}")
    
    if response.status_code == 200:
        label = response.text.strip()
        print(f"{id}: {label}")
    else:
        print(f"{id}: Not found {response.status_code}")



