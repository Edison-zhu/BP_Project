import numpy as np
import pandas as pd


def init_weights_and_bias(inputs, num_hidden_layers, hidden_neurons_list, init_method='Gaussian'):
    num_samples, feature_dim = inputs.shape
    num_neuron_list = [num_samples]
    num_neuron_list.extend(hidden_neurons_list)
    weights = {}
    bias = {}
    if init_method is None:
        for i in num_hidden_layers:
            weights[i] = np.random.randn((num_neuron_list[i], num_neuron_list[i+1]))
            bias[i] = np.zeros((num_neuron_list[i+1], 1))
    return weights, bias

def forward_pass(inputs, weights, bias, num_hidden_layers):
    for i in num_hidden_layers:
        weight = weights[i]
        bias = bias[i]
        outputs = sigmoid(np.dot(weight, inputs) + bias)s
        inputs = outputsss
    return outputs

def backward_pass():   # handling back_propagitioin here



def update_weights_and_bias():

def sigmoid(c):

    output = 1/(1+ np)

def softmax(hidden_arr):
    exp_sum = np.sum(np.exp(hidden_arr), axis=1, keepdims=True)
    output_probs = hidden_arr / exp_sum
    return output_probs

# GRADED FUNCTION: backward_propagation

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    ### START CODE HERE ### (≈ 2 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###

    # Retrieve also A1 and A2 from dictionary "cache".
    ### START CODE HERE ### (≈ 2 lines of code)
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]
    ### END CODE HERE ###

    # Backward propagation: calculate dW1, db1, dW2, db2.
    ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * (np.sum(dZ2, axis=1, keepdims=True))
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * (np.dot(dZ1, X.T))
    db1 = (1 / m) * (np.sum(dZ1, axis=1, keepdims=True))
    ### END CODE HERE ###

    # NN_model
    def nn_model(X, Y, n_h, learning_rate, num_iterations=10000, print_cost=False):
        n_x = layer_sizes(X, Y)[0]
        n_y = layer_sizes(X, Y)[2]

        # Initialize parameters
        parameters = initialize_parameters(n_x, n_h, n_y)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Loop (gradient descent)
        for i in range(0, num_iterations):
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache"
            A2, cache = forward_propagation(X, parameters)
            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost"
            cost = compute_cost(A2, Y, parameters)
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads"
            grads = backward_propagation(parameters, cache, X, Y)
            # Update rule for each parameter
            parameters = update_parameters(parameters, grads, learning_rate)
            # If print_cost=True, Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
        # Returns parameters learnt by the model. They can then be used to predict output
        return parameters

    # GRADED FUNCTION: update_parameters

    def update_parameters(parameters, grads, learning_rate):
        """
        Updates parameters using the gradient descent update rule given above

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients

        Returns:
        parameters -- python dictionary containing your updated parameters
        """
        # Retrieve each parameter from the dictionary "parameters"
        ### START CODE HERE ### (≈ 4 lines of code)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        ### END CODE HERE ###

        # Retrieve each gradient from the dictionary "grads"
        ### START CODE HERE ### (≈ 4 lines of code)
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]
        ## END CODE HERE ###

        # Update rule for each parameter
        ### START CODE HERE ### (≈ 4 lines of code)
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        ### END CODE HERE ###

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


if __name__ == '__main__':
    input_features = np.ones((10, 5))
