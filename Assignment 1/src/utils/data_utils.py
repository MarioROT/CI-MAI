import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import scipy.io
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

class Cal101Dataset(Dataset):
    def __init__(self, X, Y, transform=None, target_transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.target_transform = target_transform
        self.cat_labels = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = torch.Tensor(self.X[idx])
        label = torch.Tensor(self.Y[idx]).type(torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def readMatlabData(dir):
    mat = scipy.io.loadmat(dir)
    randSort = torch.randperm(len(mat['X']))
    X = torch.Tensor(mat['X'])[randSort]
    Y = torch.Tensor(mat['Y'][0] - 1).long()[randSort]
    encodedLabels = [mat["classnames"][0][int(y)] for y in Y]
    Y = F.one_hot(Y)
    return X,Y, encodedLabels

def splitDataset(X, Y, distribution=[0.8,0.1,0.1]):
    first_split = sum(distribution[1:])
    splitedData = iterative_train_test_split(X, Y, first_split)
 
    if len(distribution) == 3:
        second_split = distribution[2]/sum(distribution[1:])
        splitedData = splitedData[:2] + iterative_train_test_split(splitedData[2], splitedData[3], second_split)
    
    return splitedData 
