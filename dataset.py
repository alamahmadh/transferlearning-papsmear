import pandas as pd
from PIL import Image

class PapSmearDataset():
    def __init__(self, csv_path, transform=None):
        self.df_csv = pd.read_csv(csv_path)
        self.class_names = self.df_csv['class'].unique()
        self.num_classes = len(self.class_names)
        self.id_to_class = dict(zip(range(self.num_classes), self.class_names))
        self.class_to_id = dict(zip(self.class_names, range(self.num_classes)))
        self.transform = transform
        
    def __len__(self):
        return len(self.df_csv)
    
    def __getitem__(self, idx):
        file_path = self.df_csv.iloc[idx]['file_path']
        image = Image.open(file_path)
        label = self.df_csv.iloc[idx]['class']
        
        if self.transform:
            image = self.transform(image)
        
        dict_image = {
            'image': image,
            'label': self.class_to_id[label]
        }
        return dict_image
    
    def find_labels(self):
        return self.class_names