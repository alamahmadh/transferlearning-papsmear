import os
import argparse
import pandas as pd
import numpy as np

#create the parser
parser = argparse.ArgumentParser(
    description='Split the dataset into train, val, and test sets based on a metadata in the form of a csv file'
)
parser.add_argument('--file', '-f', type=str, help='The name of your csv file')
parser.add_argument('--train', '-t', type=float, default=0.7, help='the desired percentage of the train set. Default 0.7')
parser.add_argument('--val', '-v', type=float, default=0.2, help='the desired percentage of the valid set. Default 0.2')

#Parse the argument
args = parser.parse_args()

TRAIN_PERCENT = args.train
VAL_PERCENT = args.val
#df_csv = pd.read_csv(os.path.join(DATA_PATH, CSV_FILENAME))
df_csv = pd.read_csv(args.file)

SEED = 42

def main():
    all_data = []

    # each row in the CSV file corresponds to the image
    for i in range(len(df_csv)):
        img_name = df_csv.iloc[i]['file_name']
        img_label = df_csv.iloc[i]['class']
        img_dir = df_csv.iloc[i]['file_path']
    
        #if os.path.exists(img_dir):
        #    img = Image.open(img_dir)
        #    if img.mode == "RGB":
        #        all_data.append([img_name, img_label, img_dir])
                
        all_data.append([img_name, img_label, img_dir])
            
    #set the seed of the random numbers generator       
    np.random.seed(SEED)
    idxs = np.random.choice(len(all_data), len(all_data), replace=False)
    
    #construct a numpy array from the list
    all_data = np.asarray(all_data)
    num_train_files = int((len(all_data) + 1) * TRAIN_PERCENT)
    num_val_files = int((len(all_data) + 1) * VAL_PERCENT)
        
    cols = ['file_name', 'class', 'file_path']

    pd.DataFrame(all_data[idxs][:num_train_files], columns=cols).to_csv('csv_train.csv', index=False)
    pd.DataFrame(all_data[idxs][num_train_files:num_train_files+num_val_files], columns=cols).to_csv('csv_val.csv', index=False)
    pd.DataFrame(all_data[idxs][num_train_files+num_val_files:], columns=cols).to_csv('csv_test.csv', index=False)
    
    print(f'Split the data from the file {args.file} into train, val, and test sets is DONE!')
    print('Obtain three csv files corresponding to each respective set')

if __name__ == "__main__":
    main()
