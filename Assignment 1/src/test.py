from utils.data_utils import Cal101Dataset, splitDataset, readMatlabData
from utils.train_inference import trainInferenceModel
from utils.custom_plots import custom_grids
from architectures.mlp import MLP
from architectures.cnn import CNN

from torchsummary import summary
from torch.utils.data import DataLoader

import neptune.new as neptune
import os
from dotenv import load_dotenv
import argparse

load_dotenv()
USER=os.getenv("a_neptune")
USER2=os.getenv("m_neptune")
PASS=os.getenv("a_acc")
PASS2=os.getenv("m_acc")

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-ex", "--Experiment", help = "Experiment to perform CNN or MLP", default="CNN",type=str)
parser.add_argument("-ds", "--DataSplit", help = "Dataset splitting", default="0.8-0.1-0.1",type=str)
parser.add_argument("-mlp", "--MLP", help="Multi-layer perceptron architecture", default="784-50-101", type=str)
parser.add_argument("-cnn", "--CNN", help="Convolutional Neural Network architecture", default="1-5-8", type=str)
parser.add_argument("-krl", "--Kernels", help="Kernels for the cnn layers", default="3-3-3", type=str)
parser.add_argument("-ha", "--HiddenAct", help="Set hidden activation values", default='logsig', type=str)
parser.add_argument("-oa", "--OutputAct", help="Output layer activation", default="softmax", type=str)
parser.add_argument("-pl", "--PoolingL", help="Pooling layer to use", default="max_pool", type=str)
parser.add_argument("-pd", "--Padding", help="Padding to use in convoluyion", default=0, type=int)
parser.add_argument("-st", "--Stride", help="Stride for convolution and pooling", default=1, type=int)
parser.add_argument("-e", "--Epochs", help="Epochs number", default=50, type=int)
parser.add_argument("-l", "--LossFunc", help="Loss function", default = 'cce', type=str)
parser.add_argument("-o", "--Optimizer", help="Set optimizer", default='SGD', type=str)
parser.add_argument("-m", "--Momentum", help="Set momentum value", default=0.9, type=float)
parser.add_argument("-lr", "--LearningRate", help="Set learning rate", default=5e-3, type=float)
parser.add_argument("-t", "--Tolerance", help="Tolerance of unchaged epochs", default=10, type=int)
parser.add_argument("-d", "--Device", help="Device in which to run the training", default="cuda", type=str)
parser.add_argument("-tg", "--Tags", help="Adding tags to neptune project", nargs='+', default=["non-mandatory"])
 
args = parser.parse_args()

mL = lambda x, tp: list(map(tp,x.split("-")))

run = neptune.init_run(project=USER2,
                       api_token=PASS2,
                       tags = args.Tags,
                       )  

run["parameters"] = args

X,Y, encLabels = readMatlabData('../data/caltech101_silhouettes_28.mat')
X_train, Y_train, X_val, Y_val, X_test, Y_test = splitDataset(X,Y,mL(args.DataSplit, float))    

images = [sample.reshape(28,28) for sample in X]
custom_grids(images[:20], 4, 5, encLabels[:20])

train_data = Cal101Dataset(X_train, Y_train, reshape=(1,28,28) if args.Experiment == 'CNN' else None)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0)

val_data = Cal101Dataset(X_val, Y_val, reshape=(1,28,28) if args.Experiment == 'CNN' else None)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=0)

test_data = Cal101Dataset(X_test, Y_test, reshape=(1,28,28) if args.Experiment == 'CNN' else None)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=0)

model = CNN(mL(args.CNN, int), mL(args.Kernels, int), mL(args.MLP, int), args.Padding, args.Stride, args.HiddenAct, args.OutputAct, args.PoolingL) if args.Experiment == 'CNN' else MLP(mL(args.MLP, int), args.HiddenAct, args.OutputAct)

summary(model, (1,28,28) if args.Experiment == 'CNN' else (1,784), device ='cpu')

tim = trainInferenceModel(model, args.LossFunc, args.LearningRate, args.Epochs, args.Optimizer, args.Momentum, args.Tolerance, args.Device, run)
tim.train(train_dataloader, val_dataloader)
tim.test(test_dataloader,True,(4,4),16)
