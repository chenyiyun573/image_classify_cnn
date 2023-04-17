
class NetworkInfoNode:
    def __init__(self, labels_tree_dict: dict,  input_label = 'Root'):
        self.labels_tree_dict = labels_tree_dict

        self.input_label = input_label
        self.end_level = False

        self.input_id_list = []
        self.label_mapping = {}

        self.children = []
        
        self.output_label_list = []

        self.trainloader = None

        for key, value in labels_tree_dict.items():
            self.output_label_list.append(key)
            if isinstance(value, list):
                self.input_id_list.extend(value)
                child = NetworkInfoNode({}, key)
                child.input_id_list = value
                for item in value:
                    self.label_mapping[item]=key
                    child.label_mapping[item]=item
                child.output_label_list.extend(value)
                child.end_level = True
                self.children.append(child)
            elif isinstance(value, dict):
                child = NetworkInfoNode(value, key)
                self.input_id_list.extend(child.input_id_list)
                for item in child.input_id_list:
                    self.label_mapping[item]=item
                self.children.append(child)
        


    def __str__(self, level=0):
        node_str = '  ' * level + f"Node level {level}:\n" + "Network:"+ str(self.input_label) +"  Labels:"+str(self.output_label_list)+ "\n"
        for i, child in enumerate(self.children):
            node_str += f"{child.__str__(level + 1)}"
        return node_str

