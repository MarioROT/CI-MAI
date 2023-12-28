import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

import pygad
import pygad.kerasga

from evostra import EvolutionStrategy


def splitDataset(X, Y, distribution=[0.8,0.1,0.1]):
    first_split = sum(distribution[1:])
    splitedData = iterative_train_test_split(X, Y, first_split)
 
    if len(distribution) == 3:
        second_split = distribution[2]/sum(distribution[1:])
        splitedData = splitedData[:2] + iterative_train_test_split(splitedData[2], splitedData[3], second_split)
    
    return {'X_train': splitedData[0],
            'y_train': splitedData[1],
            'X_val':splitedData[2],
            'y_val':splitedData[3],
            'X_test':splitedData[4],
            'y_test':splitedData[5]} 
    
def scatter_data(data, **kargs):
    sns.scatterplot(x=data[0] , y=data[1], hue=data[2], data=data,s=100)
    plt.xlabel(kargs['var1'])
    plt.ylabel(kargs['var2'])
    plt.title(f"Scattter plot {kargs['datasetName']} Dataset")
    plt.show()
    
class GAESMLP():
    def __init__(self,
                 model,
                 optimizer,
                 pop_size = 40,
                 sigma = 0.2, 
                 learning_rate = 0.3,
                 decay = 0.995,
                 num_threads = 1,
                 num_solutions = 10,
                 num_generations = 250,
                 num_parents_mating = 5,
                 initial_population = None,
                 momentum = 0.9,
                 nesterov = True
                 ):
        
        self.model = model if not isinstance(model, list) else self.generate_model(model)
        self.optimizer = optimizer
        self.pop_size = pop_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = num_threads
        self.num_solutions = num_solutions
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.initial_population = initial_population
        self.momentum = momentum
        self.nesterov = nesterov
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
    def fit(self, X, y, **kargs):
        tf.keras.backend.clear_session()
        iterations = kargs['iterations'] if 'iterations' in kargs.keys() else 100
        verbose = kargs['verbose'] if 'verbose' in kargs.keys() else True
        
        self.X_train = X
        self.y_train = y
        if self.optimizer == 'ES':
            def fitness_func(weights):
                self.model.set_weights(weights)
                prediction = self.model.predict(self.X_train, verbose = 0)
                # reward = -self.bce(self.y_train, prediction).numpy()
                reward = 1.0 / (self.bce(self.y_train, prediction).numpy() + 0.00000001)
                return reward
            es = EvolutionStrategy([w for layer in self.model.layers for w in layer.get_weights()], fitness_func, population_size=self.pop_size, sigma=self.sigma, learning_rate=self.learning_rate, decay=self.decay, num_threads=self.num_threads)
            print_step = kargs['print_step'] if 'print_step' in kargs.keys() else 10
            es.run(iterations, print_step)
            optimized_weights = es.get_weights()
            self.model.set_weights(optimized_weights)
        elif self.optimizer == 'GA':
            def fitness_func(solution, sol_idx):
                model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=self.model,
                                                                weights_vector=solution)
                self.model.set_weights(weights=model_weights_matrix)
                predictions = self.model.predict(self.X_train, verbose=0)
                reward = 1.0 / (self.bce(self.y_train, predictions).numpy() + 0.00000001)
                return reward
            def callback_generation(ga_instance):
                print("Generation = {generation}".format(generation=ga_instance.generations_completed))
                print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
                
            weights_vector = pygad.kerasga.model_weights_as_vector(model=self.model)
            keras_ga = pygad.kerasga.KerasGA(model=self.model, num_solutions=self.num_solutions)
            
            self.initial_population = self.initial_population if self.initial_population else keras_ga.population_weights
            
            ga_instance = pygad.GA(num_generations=iterations, 
                       num_parents_mating=self.num_parents_mating, 
                       initial_population=self.initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)
            
            ga_instance.run()
            
            # Returning the details of the best solution.
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            if verbose:
                print(f"Fitness value of the best solution = {solution_fitness}")
                ga_instance.plot_result(title="MLP + GA - Iteration vs. Fitness", linewidth=4)

            # Fetch the parameters of the best solution.
            best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=self.model,
                                                                          weights_vector=solution)
            self.model.set_weights(best_solution_weights)
            
        elif self.optimizer == 'BP':
            sgd = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum, nesterov=self.nesterov)
            self.model.compile(loss = self.bce, optimizer = 'sgd', metrics = ['accuracy'])
            history = self.model.fit(self.X_train, self.y_train, epochs=iterations)
            pd.DataFrame(history.history).plot()
            plt.xlabel("epoch")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
            plt.show()
            
        if 'X_val' in kargs.keys() and 'y_val' in kargs.keys():
            acc = self.evaluate(kargs['X_val'], kargs['y_val'])
            loss = self.bce(kargs['y_val'], self.predict(kargs['X_val'])).numpy()
            if verbose:
                print(f"Validation loss: {loss}    Validation acc:  {acc}")
                self.boundary(self.model, self.X_train, self.y_train, 100, f"Train - Optimizer: {self.optimizer}")
                self.boundary(self.model, kargs['X_val'], kargs['y_val'], 100, f"Validation - Optimizer: {self.optimizer}")    
        
    def predict(self, X):
        return np.array([0.0 if i<0.5 else 1.0 for i in self.model.predict(X, verbose = 0)])
    
    def evaluate(self, X, y, verbose=None):
        preds = [0.0 if i<0.5 else 1.0 for i in self.model.predict(X, verbose = 0)]
        accuracy = accuracy_score(preds, y)
        if verbose:
            self.boundary(self.model, X, y, 100, f"{verbose} - Optimizer: {self.optimizer}")
        return accuracy
    
    @staticmethod
    def generate_model(model):
        mod = tf.keras.Sequential()
        mod.add(tf.keras.layers.Input(model[0]))
        for i in model[1:-1]:
            mod.add(tf.keras.layers.Dense(i, activation='relu'))
        mod.add(tf.keras.layers.Dense(model[-1], activation='sigmoid'))
        return mod
    
    @staticmethod
    def boundary(model, data, class_col=None, resolution=100, title=None):
        #No necesario por como se define la siguiente parte
        if class_col is not None:
            cl = class_col
        else:
            cl = 1

        k = len(np.unique(cl))
        x1=data.T[0]
        x2=data.T[1]

        # Make grid   
        x1_min, x1_max =x1.min()-0.2 , x1.max() +0.2
        x2_min ,x2_max =x2.min()-0.2 , x2.max() +0.2

        xx1, xx2 = np.meshgrid(np.linspace(x1_min,x1_max, resolution), np.linspace(x2_min, x2_max, 100))

        #Predict grid with model
        x_in = np.c_[xx1.ravel(), xx2.ravel()]
        y_pred = model.predict(x_in)
        y_pred = np.round(y_pred).reshape(xx1.shape)


        #Plotting contours
        plt.contourf(xx1, xx2, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7 )
        plt.scatter(x1, x2, c=cl, s=40, cmap=plt.cm.RdYlBu,)
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        if title:
            plt.title(title)
        plt.show()