import numpy as np
from random import shuffle
from past.builtins import xrange

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
  scores = np.dot(X, W)
  for i in xrange(X.shape[0]):
    score = scores[i,:]
    proba = np.true_divide(np.exp(score - np.max(score)), 
                           np.exp(score - np.max(score)).sum())
    loss -= np.log(proba[y[i]])
    for j in xrange(W.shape[1]):
      if j == y[i]:
        dW[:,j] += (proba[j] - 1) * X[i]
      else:
        dW[:,j] += proba[j] * X[i]
  dW = dW/X.shape[0] + reg * W
  loss = loss/X.shape[0] + 0.5 * reg * np.square(W).sum()
  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  scores = np.dot(X, W)
  scores_max = np.max(scores, axis = 1)
  proba = np.true_divide(np.exp(scores - scores_max.reshape(-1, 1)),
                         np.exp(scores - scores_max.reshape(-1, 1))
                         .sum(axis = 1).reshape(-1, 1))
  # print proba.sum()
  loss = - np.log(proba[range(X.shape[0]),y]).sum()/X.shape[0] + 0.5 * reg * np.square(W).sum()

  proba[range(X.shape[0]),y] -= 1 
  dW = np.dot(X.transpose(), proba) / X.shape[0] + reg * W
  # print proba

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

