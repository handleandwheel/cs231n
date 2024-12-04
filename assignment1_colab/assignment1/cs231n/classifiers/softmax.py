from builtins import range
import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_example = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_example):
      x = X[i, :]
      exp_scale = 0.0
      dW_tmp = np.zeros_like(W)
      for j in range(num_classes):
        w = W[:, j]
        if j == y[i]:
          loss -= w@x
          dW[:, j] -= x
        dW_tmp[:, j] = np.exp(w@x) * x
        exp_scale += np.exp(w@x)
      dW += dW_tmp / exp_scale
      loss += np.log(exp_scale)
    loss /= num_example
    dW /= num_example
    loss += reg * np.sum(W**2)
    dW += 2.0 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_example = X.shape[0]
    num_classes = W.shape[1]
    
    S = X@W
    e_S = np.exp(S)
    loss -= np.sum(S[np.arange(num_example), y])
    loss += np.sum(np.log(np.sum(e_S, axis=1)))
    loss /= num_example
    loss += reg * np.sum(W**2)

    for i in range(num_example):
      dW[:, y[i]] -= X[i, :].reshape(-1)
    
    dW += X.T @ np.diag(1/np.sum(e_S, axis=1)) @ e_S
    dW /= num_example
    dW += 2.0 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
