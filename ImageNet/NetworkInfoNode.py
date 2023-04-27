class NetworkInfoNode:
    def __init__(self, labels_tree_dict: dict):
        self.labels_tree_dict = labels_tree_dict

        self.input_label = labels_tree_dict['name']
        self.end_level = False

        self.input_id_list = []
        self.label_mapping = {}

        self.children = []

        self.output_label_list = []

        self.trainloader = None
        self.model = None

        if 'children' in self.labels_tree_dict:
            if isinstance(self.labels_tree_dict['children'][0], str):
                # children is a list of string, means that this is the deepest level
                self.input_id_list.extend(self.labels_tree_dict['children'])
                for item in self.labels_tree_dict['children']:
                    self.label_mapping[item] = item
                self.output_label_list.extend(self.labels_tree_dict['children'])
                self.end_level = True
            else:
                
                for child_dict in self.labels_tree_dict['children']:
                    self.output_label_list.append(child_dict['name'])
                    child = NetworkInfoNode(child_dict)
                    self.input_id_list.extend(child.input_id_list)
                    for item in child.input_id_list:
                        self.label_mapping[item] = child.input_label
                    self.children.append(child)

    def __str__(self, level=0):
        node_str = '  ' * level + f"Node level {level}:\n" + "Network:" + str(self.input_label) + "  Labels:" + str(
            self.output_label_list) + "\n"
        for i, child in enumerate(self.children):
            node_str += f"{child.__str__(level + 1)}"
        return node_str
