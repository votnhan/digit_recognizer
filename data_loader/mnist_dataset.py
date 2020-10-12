import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as tf

class MNISTDataset(Dataset):
    def __init__(self, data_dir, csv_file, label=True, transforms=None):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.inc_label = label
        self.data_path = os.path.join(data_dir, csv_file)
        self.dataframe = pd.read_csv(self.data_path)
        if 'Unnamed: 0' in self.dataframe.columns:
            self.dataframe.set_index('Unnamed: 0', inplace=True)
        self.len = len(self.dataframe)
        self.image_size = 28
        self.transforms = transforms

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.inc_label:
            label = self.dataframe.iloc[index, 0]
            vector_data = np.asarray(self.dataframe.iloc[index, 1:], 
                                            dtype=np.float32)
        else:
            vector_data = np.asarray(self.dataframe.iloc[index, :], 
                                            dtype=np.float32)

        data = vector_data.reshape(self.image_size, self.image_size)
        data = np.expand_dims(data, axis=2)
        
        data_tensor = data

        if self.transforms:
            data_tensor = self.transforms(data_tensor)

        if self.inc_label:
            label_tensor = torch.tensor(label)
            return data_tensor, label_tensor, self.dataframe.index[index]
        
        return data_tensor, self.dataframe.index[index]


        




    
