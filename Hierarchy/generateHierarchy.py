import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')
class_ids = [
    "n01443537", "n01944390", "n02123045", "n02281406", "n02669723", "n02841315", "n03026506", "n03404251", "n03770439", "n03983396", "n04259630", "n04486054", "n07615774", "n09246464",
    "n01629819", "n01945685", "n02123394", "n02321529", "n02699494", "n02843684", "n03042490", "n03424325", "n03796401", "n03992509", "n04265275", "n04487081", "n07695742", "n09256479",
    "n01641577", "n01950731", "n02124075", "n02364673", "n02730930", "n02883205", "n03085013", "n03444034", "n03804744", "n04008634", "n04275548", "n04501370", "n07711569", "n09332890",
    "n01644900", "n01983481", "n02125311", "n02395406", "n02769748", "n02892201", "n03089624", "n03447447", "n03814639", "n04023962", "n04285008", "n04507155", "n07715103", "n09428293",
    "n01698640", "n01984695", "n02129165", "n02403003", "n02788148", "n02906734", "n03100240", "n03544143", "n03837869", "n04067472", "n04311004", "n04532106", "n07720875", "n12267677",
    "n01742172", "n02002724", "n02132136", "n02410509", "n02791270", "n02909870", "n03126707", "n03584254", "n03838899", "n04070727", "n04328186", "n04532670", "n07734744",
    "n01768244", "n02056570", "n02165456", "n02415577", "n02793495", "n02917067", "n03160309", "n03599486", "n03854065", "n04074963", "n04356056", "n04540053", "n07747607",
    "n01770393", "n02058221", "n02190166", "n02423022", "n02795169", "n02927161", "n03179701", "n03617480", "n03891332", "n04099969", "n04366367", "n04560804", "n07749582",
    "n01774384", "n02074367", "n02206856", "n02437312", "n02802426", "n02948072", "n03201208", "n03637318", "n03902125", "n04118538", "n04371430", "n04562935", "n07753592",]



def find_hypernym_paths(synset):
    hypernym_paths = synset.hypernym_paths()
    return [[s.name().split(".")[0] for s in path] for path in hypernym_paths]

def main():
    class_hierarchy = {}

    for class_id in class_ids:
        synset = wn.synset_from_pos_and_offset('n', int(class_id[1:]))
        class_name = synset.name().split(".")[0]
        hypernym_paths = find_hypernym_paths(synset)

        # Choose the longest hypernym path to represent the class hierarchy
        longest_path = sorted(hypernym_paths, key=lambda x: len(x), reverse=True)[0]
        longest_path = longest_path[1:]  # Remove 'entity' from the path
        
        current_level = class_hierarchy

        for category in longest_path:
            if category not in current_level:
                current_level[category] = {}
            current_level = current_level[category]

    return class_hierarchy


def get_synset_hierarchy(synset):
    hypernyms = synset.hypernyms()
    if not hypernyms:
        return [synset]
    else:
        hierarchy = []
        for hypernym in hypernyms:
            hierarchy.extend(get_synset_hierarchy(hypernym))
        hierarchy.append(synset)
        return hierarchy

class_hierarchy = {}
for class_id in class_ids:
    synset = wn.synset_from_pos_and_offset('n', int(class_id[1:]))
    hierarchy = get_synset_hierarchy(synset)
    
    current_level = class_hierarchy
    for syn in hierarchy[:-1]:
        name = syn.name().split('.')[0]
        if name not in current_level:
            current_level[name] = {}
        current_level = current_level[name]
    current_level[hierarchy[-1].name().split('.')[0]] = class_id

print(class_hierarchy)


if __name__ == "__main__":
    # hierarchy = main()
    # print(hierarchy)
    pass
