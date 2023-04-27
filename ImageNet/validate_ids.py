import json
from NetworkInfoNode import NetworkInfoNode

with open("tree_structure.json", "r") as infile:
    labels_tree_dict = json.load(infile)

root_network_info_node = NetworkInfoNode(labels_tree_dict)

# Print root node information
# print(root_network_info_node.__str__())

# Print information for each child (second level) and their respective children (third level)
for child in root_network_info_node.children:
    print(child.input_label)
    print(child.label_mapping)
    for child1 in child.children:
        # print(child1.input_label)
        print(child1.input_label)
        print(child1.label_mapping)
    break
    # print(child.input_id_list)

#print(root_network_info_node.label_mapping)
