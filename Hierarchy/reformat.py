import os
import shutil
from constants import class_hierarchy
from constants import dataset_path

# define source and destination directories
src_dir = dataset_path+'tiny-imagenet-200/train'
dst_dir = dataset_path+'re-tiny-imagenet-200/new_train'

id_dict = class_hierarchy
# create a reverse id_dict from the id_dict
id_dict_rev = {}
for class_name, ids in id_dict.items():
    for id_ in ids:
        id_dict_rev[id_] = class_name

# iterate over the files in the source directory
for root, dirs, files in os.walk(src_dir):
    for filename in files:
        if filename.endswith('.JPEG'):
            # get the class id and class name from the file name
            file_id = filename.split('_')[0]
            class_name = id_dict_rev[file_id]

            # construct the destination directory path
            dst_subdir = os.path.join(dst_dir, class_name)

            # create the destination directory if it doesn't exist
            if not os.path.exists(dst_subdir):
                os.makedirs(dst_subdir)

            # copy the file to the destination directory
            src_path = os.path.join(root, filename)
            dst_path = os.path.join(dst_subdir, filename)
            shutil.copy(src_path, dst_path)
