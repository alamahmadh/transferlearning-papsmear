# Transfer learning approach for the cell classification on pap smear images
This repository is aimed to document some simple codes for the data preprocessing and transfer learning approach that I had used for the pap smear classification.

## Split the data into the train, validation, and test sets
We have created a csv file that store metadata information about the images (name of image, cell classification, and image path). The example can be seen in `papsmear_metadata_snippet.csv`, which is a snippet of the data we used for this task. Using `split_data.py` in `samples` to create three separate csv files from the main csv file that each corresponds to a respective set.

`python samples/split_data.py --file <csv filename> --train <percentage of train set> --val <percentage of val set>` 

Default usage (70% train, 20% val, 10% test): `python samples/split_data.py -f papsmear_metadata_snippet.csv` 
