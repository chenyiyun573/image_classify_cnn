import json
import os
import random
from nltk.corpus import wordnet as wn

# Get ImageNet class ids from folder names
imagenet_train_path = "/mnt/imagenet/ILSVRC2012_img_train_extracted"
imagenet_class_ids = [folder_name for folder_name in os.listdir(imagenet_train_path) if os.path.isdir(os.path.join(imagenet_train_path, folder_name))]

# Get WordNet synsets for ImageNet classes
synsets = [wn.synset_from_pos_and_offset('n', int(offset[1:])) for offset in imagenet_class_ids]

# Shuffle synsets
random.shuffle(synsets)

# Create tree structure
root_node = {
    "name": "Root",
    "children": []
}

num_classes_per_network = len(synsets) // 20

for i in range(5):
    second_level_node = {
        "name": f"Network {i+1}",
        "children": []
    }

    for j in range(4):
        third_level_node = {
            "name": f"Network {i+1}-{j+1}",
            "children": []
        }

        # Distribute synsets among third level nodes
        for _ in range(num_classes_per_network):
            if synsets:
                synset = synsets.pop()
                class_id = f"n{synset.offset():08d}"
                third_level_node["children"].append(class_id)
            else:
                break

        second_level_node["children"].append(third_level_node)

    root_node["children"].append(second_level_node)

# Output the tree structure in JSON format
with open("tree_structure.json", "w") as outfile:
    json.dump(root_node, outfile, indent=2)
