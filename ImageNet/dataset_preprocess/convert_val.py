import os
import shutil
import pandas as pd
from scipy.io import loadmat

# Load the meta data
meta_data = loadmat("/mnt/imagenet/ILSVRC2012_devkit/ILSVRC2012_devkit_t12/data/meta.mat")
synsets = meta_data['synsets']
labels = [str(synset[0][1][0]) for synset in synsets]

# Load the validation ground truth
val_ground_truth = pd.read_csv(
    "/mnt/imagenet/ILSVRC2012_devkit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
    header=None,
    names=['label']
)

# Create the required folder structure
src_folder = "/mnt/imagenet/ILSVRC2012_img_val"
dst_folder = "/mnt/imagenet/ILSVRC2012_img_val_extracted"
os.makedirs(dst_folder, exist_ok=True)

for idx, row in val_ground_truth.iterrows():
    label = labels[row['label'] - 1]
    label_folder = os.path.join(dst_folder, label)
    os.makedirs(label_folder, exist_ok=True)
    src_file = os.path.join(src_folder, f"ILSVRC2012_val_{idx + 1:08}.JPEG")
    dst_file = os.path.join(label_folder, f"ILSVRC2012_val_{idx + 1:08}.JPEG")
    shutil.move(src_file, dst_file)

print("Validation dataset extracted successfully.")