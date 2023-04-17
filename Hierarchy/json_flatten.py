import json

# Load your hierarchy JSON file
with open('input_dict.json', 'r') as f:
    input_json = json.load(f)

def flatten_json(json_data):
    result = []
    if isinstance(json_data, list):
        result.extend(json_data)
        return result

    
    for key, value in json_data.items():
        if isinstance(value, dict):
            result.extend(flatten_json(value))
        elif isinstance(value, list):
            result.extend(value)
    
    return result

sub_networks = {
    'Animals': [],
    'Objects': [],
    'Others': []
}

for category, content in input_json.items():
    if category in sub_networks:
        sub_networks[category] = flatten_json(content)
    else:
        sub_networks['Others'].extend(flatten_json(content))

# Write the result to a JSON file
with open('output.json', 'w') as outfile:
    json.dump(sub_networks, outfile, indent=2)
