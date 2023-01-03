# Transfer learning approach for the cell classification on pap smear images
This repository is aimed to document some simple codes for the data preprocessing and transfer learning approach that I had used for the pap smear classification.

## Split the data into the train, validation, and test sets
We have created a csv file that store metadata information about the images (name of image, cell classification, and image path). The example can be seen in `papsmear_metadata_snippet.csv`, which is a snippet of the data we used for this task. Using `split_data.py` in `samples` to create three separate csv files from the main csv file that each corresponds to a respective set.

`python samples/split_data.py --file <csv filename> --train <percentage of train set> --val <percentage of val set>` 

Default usage (70% train, 20% val, 10% test): `python samples/split_data.py -f papsmear_metadata_snippet.csv` 

## Start training a model
The configuration file located in `config/config.py` can be changed accordingly. For example, in that file we employed a pretrained `resnet18` model with `num_epochs=4` and so on. 

Default Usage: `python train.py`

The best model will be saved in `models/checkpoints`. This pipeline of the classification task is far from optimal (the evaluation metric is still poor) since this approach simply unfreeze the last linear layer (the classifier) and it is strongly recommended for the whole model to be fine-tuned instead. I will work further on different transfer learning strategies to optimize the classification task (not only the layer freezing or whatever but also using more recent and advanced network architectures). In addition, I need to elaborate on the more proper evaluation metrics to asses the models.

This work is still in progress. Hopefully I would be able to improve this in near future (of course, this is for my own documentation so that I can recycle and improve this template of DL pipeline for another future projects)

## Inference stage
Upcoming...
