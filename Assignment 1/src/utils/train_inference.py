import torch
from utils.custom_plots import custom_grids
import torch.nn.functional as F
import numpy as np


class trainInferenceModel():

    def __init__(self,
                 model,
                 loss_function,
                 learning_rate,
                 epochs,
                 optimizer,
                 momentum=None,
                 tolerance = 5,
                 device='cuda',
                 run = None
                 ):

        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.momentum = momentum
        self.tolerance = tolerance
        self.device = device
        self.run = run

        settings = {'Adam':torch.optim.Adam(self.model.parameters(), lr=self.learning_rate),
                    'SGD':torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum),
                    'cce':torch.nn.CrossEntropyLoss(),
                    'mse':torch.nn.MSELoss()}

        self.loss_function = settings[loss_function] 
        self.optimizer = settings[optimizer]

        # Set the GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.device == 'cuda' else 'cpu')

        # Move the model to the GPU
        self.model = self.model.to(self.device)
        print('Device used: ', self.device)

    def train(self, train_dataloader, val_dataloader):
        print("--------------------------------------------\n TRAINING MODEL ")
        print("--------------------------------------------\n")
        self.model.train()
        old_acc = 0
        no_improvement = 0
        # Run the training loop
        for epoch in range(self.epochs): 
            # Print epoch
            print(f'Starting epoch {epoch+1}')

            # Set current loss value
            current_loss = 0.0
            current_acc = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(train_dataloader, 0):
                # Get inputs
                inputs, targets = data

                # Moving data to GPU
                inputs, targets = inputs.to(self.device), targets.to(self.device)  

                # Zero the gradients
                self.optimizer.zero_grad()

                # Perform forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.loss_function(outputs, targets)

                 # Obtain the predicted class
                if self.model.output_activation != 'softmax':
                      y_pred = torch.argmax(outputs, 1)
                else:
                      y_pred = torch.argmax(F.softmax(outputs, 1),1)

                # Compute real class
                y_true_class = torch.argmax(targets, 1)
                
                # Compute accuracy
                acc = (y_true_class == y_pred).type(torch.float32).mean()

                # Perform backward pass - Backpropagation
                loss.backward()

                # Perform optimization
                self.optimizer.step()

                # Print statistics
                current_loss += loss.item()
                current_acc += acc.item() * 100

                if self.run:
                    self.run["train/loss"].log(loss)
                    self.run["train/accuracy"].log(acc)

                if i % 100 == 99:
                      print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 100))
                      current_loss = 0.0

            # Don't compute gradients to do the validation
            with torch.no_grad():

                # Model in evaluation setup
                self.model.eval()
                
                losses, accs = [], []
                # Validation step
                for x, y_true in val_dataloader:
                    #GPU
                    x = x.to(self.device)
                    y_true = y_true.to(self.device)

                    # Fordward pass => Getting the logits
                    outputs = self.model(x)

                    # Obtain the predicted class
                    if self.model.output_activation != 'softmax':
                        y_pred = torch.argmax(outputs, 1)
                    else:
                        y_pred = torch.argmax(F.softmax(outputs, 1),1)

                    # Compute real class
                    y_true_class = torch.argmax(y_true, 1)
                      
                    # Compute loss
                    loss = self.loss_function(outputs, y_true)

                    # Compute accuracy
                    acc = (y_true_class == y_pred).type(torch.float32).mean()

                    # Save records
                    losses.append(loss.item())
                    accs.append(acc.item() * 100)

                # Print metrics
                loss = np.mean(losses)
                acc = np.mean(accs)

                if self.run:
                    self.run["validation/loss"].log(loss)
                    self.run["validation/accuracy"].log(acc)

                # Stop if there is no improvement in 5 epochs.
                if acc <= old_acc:
                      no_improvement += 1
                else:
                      no_improvement = 0
                      old_acc = acc
                      
                if no_improvement >= self.tolerance:
                      print("Convergence epoch ",epoch-self.tolerance)
                      break
                
                print(f'E{epoch+1:2} loss={loss:6.2f} acc={acc:.2f}')

        # Process is complete.
        print('Training process has finished.')

        return self.model


    def test(self, test_dataloader, viz = True, layout = None, num_viz = None):
        # Test
        # Set to evaluation mode
        self.model.eval()

        # Block gradient computations
        with torch.no_grad(): 
                
            accs = []
            for i,(x, y_true) in enumerate(test_dataloader):

                x = x.to(self.device)
                y_true = y_true.to(self.device)
                
                # forward pass
                outputs = self.model(x)
                
                # Obtain the predicted class
                if self.model.output_activation != 'softmax':
                    y_pred = torch.argmax(outputs, 1)
                else:
                    y_pred = torch.argmax(F.softmax(outputs, 1),1)

                # compute real class
                y_true_class = torch.argmax(y_true, 1)
                
                # compute loss
                loss = self.loss_function(outputs, y_true)

                # compute accuracy
                acc = (y_true_class == y_pred).type(torch.float32).mean()

                # save records
                accs.append(acc.item() * 100)

                #logging data to neptune
                if self.run:
                    self.run["test/accuracy"].log(acc)

                if i==0:
                    x_viz=x
                    y_true_class_viz=y_true_class
                    y_pred_viz=y_pred
            
            acc = np.mean(accs)

        print("------------------------------------------------------------\n")
        print(f'Test accuracy = {acc:.2f}')
        print("\n------------------------------------------------------------\n")

        if viz:
            self.visualize(x_viz, y_true_class_viz, y_pred_viz, layout, num_viz)

    def visualize(self,x, y_true_class, y_pred, layout, num_viz):
        if not num_viz:
            num_viz = len(x)
        if not layout:
            layout = [(i,j) for i in range(num_viz) for j in range(num_viz) if i*j == num_viz]
            layout = layout[int(len(layout)/2)]

        x_ant = x
        y_true_ant = y_true_class
        y_pred_ant = y_pred

        x = x_ant[:num_viz].squeeze().cpu().numpy().reshape((x_ant.shape[0],28,28))
        y_true_class = y_true_ant[:num_viz].cpu().numpy()
        y_pred = y_pred_ant[:num_viz].cpu().numpy()

        titles = [f'V={t} P={p}' for t, p in zip(y_true_class, y_pred)]
        custom_grids(x, layout[0], layout[1], titles, 'Predictions sample')
        if self.run:
            custom_grids(x, layout[0], layout[1], titles, 'Predictions sample', upload='Predictions', run=self.run)
