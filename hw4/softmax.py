import numpy as np
from layers import *


class SoftmaxClassifier:
    """
    A fully-connected neural network with softmax loss that uses a modular
    layer design.

    We assume an input dimension of D, a hidden dimension of H,
    and perform classification over C classes.
    The architecture should be fc - relu - fc - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100,
                 num_classes=10, weight_scale=1e-3):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        self.params = {"W1": np.random.normal(loc=0, scale=weight_scale, size=(input_dim, hidden_dim)),
                       "b1": np.zeros(hidden_dim),
                       "W2": np.random.normal(loc=0, scale=weight_scale, size=(hidden_dim, num_classes)),
                       "b2": np.zeros(num_classes)}
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params. Use keys 'W1' and 'b1' for the weights and       #
        # biases of the first fully-connected layer, and keys 'W2' and 'b2' for    #
        # the weights and biases of the output affine layer.                       #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Inputs:
        - X: Array of input data of shape (N, d_in)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        fc1, cache_fc1 = fc_forward(X, self.params["W1"], self.params["b1"])
        rel1, cache_relu = relu_forward(fc1)
        fc2, cache_fc2 = fc_forward(rel1, self.params["W2"], self.params["b2"])
        scores = fc2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}

        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k].                                                          #
        ############################################################################
        loss, dx_sf = softmax_loss(scores, y)
        dx_fc2, grads['W2'], grads['b2'] = fc_backward(dx_sf, cache_fc2)
        dx_relu = relu_backward(dx_fc2, cache_relu)
        dx_fc1, grads['W1'], grads['b1'] = fc_backward(dx_relu, cache_fc1)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
