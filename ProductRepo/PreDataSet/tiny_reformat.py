import os
import shutil

base_path = '.'

# Set the paths to the Tiny ImageNet train and validation directories
train_dir = base_path+'/tiny-imagenet-200/train'
val_dir = base_path+'/tiny-imagenet-200/val'

# Set the paths to the output train and validation directories with the new structure
train_output_dir = base_path+'/re-tiny-imagenet-200/train'
val_output_dir = base_path+'/re-tiny-imagenet-200/val'

# Check if the output directories exist, and create them if they don't
if not os.path.exists(train_output_dir):
    os.makedirs(train_output_dir)
if not os.path.exists(val_output_dir):
    os.makedirs(val_output_dir)

# Create a dictionary mapping class IDs to class names
class_dict = {}
with open(base_path+'/tiny-imagenet-200/wnids.txt', 'r') as f:
    for line in f:
        class_id = line.strip()
        class_dict[class_id] = []

# Populate the dictionary with image filenames for each class
for subdir, _, filenames in os.walk(train_dir):
    class_id = subdir.split('/')[-2]
    if class_id in class_dict:
        class_dict[class_id] += [os.path.join(subdir, filename) for filename in filenames if filename.endswith('.JPEG')]

# Copy the train images to the output directory with new folder structure
for class_id, filenames in class_dict.items():
    output_class_dir = os.path.join(train_output_dir, class_id)
    os.makedirs(output_class_dir, exist_ok=True)
    for filename in filenames:
        output_filename = os.path.join(output_class_dir, os.path.basename(filename))
        shutil.copyfile(filename, output_filename)

# Copy the validation images to the output directory with new folder structure
with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
    for line in f:
        filename, class_id, *_ = line.strip().split('\t')
        input_filename = os.path.join(val_dir, 'images', filename)
        output_dir = os.path.join(val_output_dir, class_id)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, filename)
        shutil.copyfile(input_filename, output_filename)

