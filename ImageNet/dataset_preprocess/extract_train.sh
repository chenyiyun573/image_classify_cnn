mkdir ILSVRC2012_img_train_extracted
for f in ILSVRC2012_img_train/*.tar; do
  folder_name=$(basename "$f" .tar)
  mkdir "ILSVRC2012_img_train_extracted/$folder_name"
  tar -xvf "$f" -C "ILSVRC2012_img_train_extracted/$folder_name"
done
