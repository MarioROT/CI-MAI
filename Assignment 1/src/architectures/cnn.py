import torch
from torch import nn
from torchvision import transforms
from torchsummary import summary
import torch.nn.functional as F


class CNN(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, CNN_arch,CNN_Kernels,MLP_arch, padding=0, stride=1, hidden_activation="logsig", output_activation='softmax',pool_layer="avg_pool"):
    super().__init__()
    
    self.CNN_arch=CNN_arch
    self.CNN_Kernels=CNN_Kernels
    self.MLP_arch=MLP_arch
    self.padding=padding
    self.stride=stride
    self.hidden_activation=hidden_activation
    self.output_activation=output_activation
    self.pool_layer=pool_layer
    
    self.settings = {"logsig":nn.LogSigmoid,
                "softmax":nn.LogSoftmax,
                "relu":nn.ReLU,
                "cat_cross_entropy":F.cross_entropy,
                "mse_loss":F.mse_loss,
                "avg_pool":nn.AvgPool2d,
                "max_pool":nn.MaxPool2d }

    layers_mlp = [layer(h_units, self.MLP_arch[i+1]) if layer ==  nn.Linear else layer() for i,h_units in enumerate(self.MLP_arch[:-1]) for layer in [nn.Linear, self.settings[self.hidden_activation]]]
    
    
    # layers_conv = [layer(h_units, self.CNN_arch[i+1],self.CNN_Kernels[i],self.padding,self.stride) if layer ==  nn.Conv2d else layer() for i,h_units in enumerate(self.CNN_arch[:-1]) for layer in [nn.Conv2d, self.settings[self.hidden_activation], self.settings[self.pool_layer]]] 

    layers_conv=[]
    for i,h_units in enumerate(self.CNN_arch[:-1]):
        for layer in [nn.Conv2d, self.settings[self.hidden_activation], self.settings[self.pool_layer]]:
            
            if layer ==  nn.Conv2d:
                # layers_conv.append(layer(h_units, self.CNN_arch[i+1],self.CNN_Kernels[i],self.padding,self.stride) )
                
                layers_conv.append(layer(h_units, self.CNN_arch[i+1],self.CNN_Kernels[i],self.stride,self.padding) )
           
            elif layer==self.settings[self.pool_layer]:
                layers_conv.append(layer(self.CNN_Kernels[i],1))
                
            else: 
                layers_conv.append(layer())
    
    
    self.layers = nn.Sequential(
      # nn.Flatten(),
      *[*layers_conv,nn.Flatten(),*layers_mlp[:-1]]
      
    )
    
    # if self.settings[self.output_activation] != 'softmax':
    #   self.out_activ = self.settings[self.output_activation]()

    if self.settings[self.output_activation] != 'softmax':
        self.out_activ = self.settings[self.output_activation]()
      
  def forward(self, x):
    '''Forward pass'''
    if self.settings[self.output_activation] != 'softmax':
      return self.out_activ(self.layers(x))
    else:
      return self.layers(x)

  def predict(self, x):
        y_prob = self.out_activ(self.layers(x))
        return y_prob
