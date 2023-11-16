import torch
from torch import nn
from torchvision import transforms
from torchsummary import summary
import torch.nn.functional as F



class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, architecture, hidden_activation, output_activation = 'softmax'):
    super().__init__()

    settings = {"logsig":nn.LogSigmoid,
                "softmax":nn.LogSoftmax,
                "relu":nn.ReLU,
                "cat_cross_entropy":F.cross_entropy,
                "mse_loss":F.mse_loss}

    layers = [layer(h_units, architecture[i+1]) if layer ==  nn.Linear else layer() for i,h_units in enumerate(architecture[:-1]) for layer in [nn.Linear, settings[hidden_activation]]]

    self.layers = nn.Sequential(
      # nn.Flatten(),
      *layers
    )
    if settings[params['output_activations']] != 'softmax':
      self.out_activ = settings[params[output_activation]]()


  def forward(self, x):
    '''Forward pass'''
    if settings[params['output_activations']] != 'softmax':
      return self.out_activ(self.layers(x))
    else:
      return self.layers(x)

  def predict(self, x):
        y_prob = self.out_activ(self.layers(x))
        return y_prob