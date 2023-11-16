from utils.data_utils import Cal101Dataset, splitDataset, readMatlabData
from utils.custom_plots import custom_grids
import numpy as np
import matplotlib.pyplot as plt

X,Y, encLabels = readMatlabData('../data/caltech101_silhouettes_28.mat')
X_train, Y_train, X_val, Y_val, X_test, Y_test = splitDataset(X,Y)

images = [sample.reshape(28,28) for sample in X]
custom_grids(images[:20], 4, 5, encLabels[:20])

train_data = Cal101Dataset(X_train, Y_train)

