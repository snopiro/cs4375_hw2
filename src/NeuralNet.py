#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile)




    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        self.processed_data = self.raw_input

        # Converting categorical columns to numerical values
        le = LabelEncoder()
        for column in self.processed_data.columns:
            if self.processed_data[column].dtype == 'object':
                self.processed_data[column] = le.fit_transform(self.processed_data[column])
        
        # Standardizing the data
        scaler = StandardScaler()
        self.processed_data = pd.DataFrame(scaler.fit_transform(self.processed_data), columns=self.processed_data.columns)

        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        #   metrics

        plt.figure(figsize=(10, 6))
        results = []

        for activation in activations:
            for lr in learning_rate:
                for max_iter in max_iterations:
                    for num_hidden in num_hidden_layers:
                        hidden_layer_sizes = tuple(100 for _ in range(num_hidden))

                        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                                             activation=activation, 
                                             learning_rate_init=lr, 
                                             max_iter=max_iter,
                                             random_state=42)
                        
                        model.fit(X_train, y_train)
                        
                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)
                        
                        train_mse = mean_squared_error(y_train, y_train_pred)
                        test_mse = mean_squared_error(y_test, y_test_pred)
                        
                        # print(f"Model with activation={activation}, learning_rate={lr}, max_iterations={max_iter}, num_hidden_layers={num_hidden}")
                        # print(f"Training MSE: {train_mse}")
                        # print(f"Test MSE: {test_mse}")

                        results.append({
                            'Activation': activation,
                            'Learning Rate': lr,
                            'Max Iterations': max_iter,
                            'Num Hidden Layers': num_hidden,
                            'Training MSE': train_mse,
                            'Test MSE': test_mse
                        })
                        
                        plt.plot(model.loss_curve_, label=f"{activation}, {lr}, {max_iter}, {num_hidden}")

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Model Loss across Iterations')
        plt.legend()

        # Save the plot to a file
        plt.savefig("files/neural_network_loss.png")

        plt.show()


        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Display the results
        print(results_df.to_string(index=False))

        # Write the results to a file
        log_file = open('files/neural_network_results.log', 'w')
        log_file.write(results_df.to_string(index=False))
        log_file.close()

        return 0




if __name__ == "__main__":
    neural_network = NeuralNet("https://personal.utdallas.edu/~cwk200000/files/auto-mpg-training.csv") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
