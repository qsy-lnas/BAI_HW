import torch 
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class EmDataset(Dataset):
    def __init__(self, split = "train"):
        super().__init__()
        data_pd = pd.read_csv('project2/figure.csv')
        data = data_pd.to_numpy()
        labels = data[:, 0]
        usage = data[:, 1]
        features = data[:, 2:]
        if split == "train":
            index = usage == "Training"
        elif split == "test":
            index = usage == "Test"
        self.label = labels[index]
        self.feature = features[index]

        #print(self.feature.shape)

    def __len__(self):
        return self.label.shape[0]

    def __get_item__(self, i):

        return self.feature[i], self.label[i]
        


if __name__ == "__main__":
    e = EmDataset()
