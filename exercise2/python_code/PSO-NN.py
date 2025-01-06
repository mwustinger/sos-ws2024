import numpy as np
import pyswarms as ps
from commonsetup import n_hidden, X_train, X_test, y_train, y_test, n_inputs, n_classes, activation, n_iteration
import itertools
import matplotlib.pyplot as plt
import time

class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_classes, activation):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.activation = activation

    def count_param(self):
        """Calculate the total number of parameters to optimize."""
        return (self.n_inputs * self.n_hidden) + (self.n_hidden * self.n_classes) + self.n_hidden + self.n_classes

    def generate_logits(self, x, data):
        """ 
        Parameters:
        x: one PSO slution (a list of variables, i.e. coordinates of a particle) 
        data: The train or test data to be predited

        At first, the function builds the network by cutting the values in x into the weights and 
        biases of the NN. Then it passes the data and performs activation to get the logits. 
        """
        ind1 = self.n_inputs * self.n_hidden
        W1 = x[0:ind1].reshape((self.n_inputs, self.n_hidden))
        ind2 = ind1 + self.n_hidden
        b1 = x[ind1:ind2].reshape((self.n_hidden,))
        ind3 = ind2 + self.n_hidden * self.n_classes
        W2 = x[ind2:ind3].reshape((self.n_hidden, self.n_classes))
        b2 = x[ind3:ind3 + self.n_classes].reshape((self.n_classes,))
        
        z1 = data.dot(W1) + b1  # Pre-activation in Layer 1
        a1 = self.activation(z1)  # Activation in Layer 1
        logits = a1.dot(W2) + b2  # Pre-activation in Layer 2
        return logits

    def forward_prop(self, params, X_train, y_train):
        """Calculate the loss using forward propagation."""
        logits = self.generate_logits(params, X_train)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Softmax
        correct_logprobs = -np.log(probs[range(X_train.shape[0]), y_train])
        loss = np.sum(correct_logprobs) / X_train.shape[0]
        return loss    
    
    def predict(self, weights, data):
        """Predict the classes based on trained weights."""
        logits = self.generate_logits(weights, data)
        return np.argmax(logits, axis=1)

class PSOOptimizer:
    def __init__(self, nn, c1, c2, w, swarm_size, n_iterations, batchsize):
        self.nn = nn
        self.c1 = c1  # self confidence
        self.c2 = c2  # swarm confidence
        self.w = w # inertia (omega)
        self.swarm_size = swarm_size
        self.n_iterations = n_iterations
        self.batchsize = batchsize

    def fitness_function(self, X, X_train, y_train):
        """
        Parameters:
        X: 2-D array holding the PSO solutions to be evaluated by the fitness function
        X_train: Train set
        Y_train: target of the train set

        This is the fitness function used by the PSO, which is to be implemented (completed) 
        by the students. The objective is understanding the concept of how PSO is applied in 
        this use case, namely optimizing a NN.

        Note that in each iteration of the PSO algorithm, a set of solutions are generated, 
        namely one solution by each particle. These are passed to this fitness function in the 
        parameter X. Since each solution is a list of numbers (the coordinates of a particle 
        position), X is a two-dimensional array.

        Note that each solution is used to setup the weights and biases of the network. 
        Therefore, what you should do here is performing the forward propagation each time
        using the solution and random batches of training data to return the resulting accuracies 
        in a one-dimensional list.

        To do this successfully, refer to and understand the functions forward_prop() and 
        generate_logits() that is called inside it.

        Note that the current implementation of the function is random, which will run
        but it will result in low accuracy ~ 1/n_classes.

        Returns:
        A 1-D array of fitness scores for each particle's solution.
        """
        
        fittness_scores = []

        for particle_solution in X:
            X_batch, y_batch = self.random_batch(X_train, y_train)
            loss = self.nn.forward_prop(particle_solution, X_batch, y_batch)
            fittness = loss     # Typically the fittness would be the inverse to the loss, higher fittness = lower loss, however, since pyswarms optimizes to the minimum, a lower function value is prefered
            fittness_scores.append(fittness)

        return np.array(fittness_scores)

    def random_batch(self, X_train, y_train):
        indices = np.random.choice(len(X_train), size=self.batchsize, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]
        return (X_batch, y_batch)
    
    def optimize(self, X_train, y_train):
        """Perform the PSO optimization."""
        dimensions = self.nn.count_param()
        optimizer = ps.single.GlobalBestPSO(n_particles=self.swarm_size, dimensions=dimensions,
                                            options={'c1': self.c1, 'c2': self.c2, 'w': self.w})
        cost, weights = optimizer.optimize(self.fitness_function, iters=self.n_iterations, verbose=True,
                                       X_train=X_train, y_train=y_train)
        
        self.loss_history = optimizer.cost_history

        return weights

def main():
    ####### PSO  Tuning ################
    # Tune the PSO parameters here trying to outperform the classic NN 
    # For more about these parameters, see the lecture resources
    par_C1 = 0.5 # 0.1 # self confidence
    par_C2 = 0.3 # 0.1 # global confidence
    par_W = 0.9  # 0.1 # inertia value
    par_SwarmSize = 30 #100
    batchsize = 100 # 200 # The number of data instances used by the fitness function

    best_accuracy = 0
    best_params = {}
    params_acc = []

    params = {
        "C1": [1.5],
        "C2": [1.0],
        "W": [0.75],
        "SwarmSize": [500]
    }

    keys, values = zip(*params.items())
    permutations_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param in permutations_dict:
        print ("############ you are using the following settings:")
        print ("Number hidden layers: ", n_hidden)
        print ("activation: ", activation[0])
        print ("Number of variables to optimize: ", (n_inputs * n_hidden) + (n_hidden * n_classes) + n_hidden + n_classes)
        print ("PSO parameters C1: ", param["C1"], "C2: ", param["C2"], "W: ", param["W"], "Swarmsize: ", param["SwarmSize"],  "Iterations: ", n_iteration)
        print ("\n")

        loss_history = []
        avg_acc = []
        times = []
        for i in range(20):
            print(f"Epoch {i}")
            start = time.time()
            # Initialize Neural Network and PSO optimizer
            nn = NeuralNetwork(n_inputs, n_hidden, n_classes, activation[0])
            pso = PSOOptimizer(nn, param["C1"], param["C2"], param["W"], param["SwarmSize"], n_iteration, batchsize)
            
            # Perform optimization
            weights = pso.optimize(X_train, y_train)

            end = time.time()
            times.append(end - start)

            loss_history.append(pso.loss_history)

            # Evaluate accuracy on the test set
            y_pred = nn.predict(weights, X_test)
            accuracy = (y_pred == y_test).mean()
            print(f"Accuracy PSO-NN: {accuracy:.2f}")
            avg_acc.append(accuracy)

        avg_acc = np.mean(avg_acc)
        avg_train_time = np.mean(times)
        carry_over = param
        carry_over["acc"] = avg_acc
        params_acc.append(carry_over)
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            best_params = param 

    carry_over = best_params
    carry_over["best_acc"] = best_accuracy
    params_acc.append(carry_over)

    plt.figure(figsize=(10, 6))

    # Plot each run as a separate line
    for i, loss in enumerate(loss_history):
        plt.plot(loss, color='black', alpha=0.5, linewidth=2)

    # Add labels, title, and legend
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curves for Multiple Runs')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.savefig('./exercise2/plots/Convergence_PSO_glass.png', dpi=300, bbox_inches='tight')

    print(f"Best Accuracy PSO-NN: {best_accuracy:.2f}")
    print(f"Best Parameters: {params}")
    print(f"Average Train Time: {avg_train_time:.2f}")

    """ with open("./exercise2/python_code/grid_search_glass.txt", 'w') as file:
        for item in params_acc:
            file.write(str(item) + '\n') """


if __name__ == "__main__":
    main()
