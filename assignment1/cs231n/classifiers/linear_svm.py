import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    summed_indicator_functions = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # Gradient w.r.t to weights of incorrect class labels
        dW[:, j] += X[i]
        # Keep count of how many times margin > 0
        summed_indicator_functions += 1

    # Gradient w.r.t. to weights of correct class label
    dW[:, y[i]] -= summed_indicator_functions*X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add L2 regularization to the loss.
  loss += reg * np.sum(W * W)
  # Regularization gradient
  dW += 2 * reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  # Compute scores
  scores = X.dot(W)
  # Get correct class scores for each training example as 1D vector
  correct_class_score = scores[np.arange(num_train), y].reshape(-1, 1)
  # Compute margin
  margin = np.maximum(0., scores-correct_class_score+1)
  # For each training example, zero margin of correct class (loss is a sum over incorrectly labeled classes)
  margin[np.arange(num_train), y] = 0
  # Compute loss
  loss = np.sum(margin)/num_train + reg * np.sum(W * W)
  # Indicator function
  margin[margin > 0] = 1
  # Count number of times margin > 1 for each training example in order to compute the gradient of correctly labeled classes
  margin[np.arange(num_train), y] = -np.sum(margin, axis=1)
  # Compute gradient
  dW = np.matmul(margin.T, X).T
  # Add regularization gradient and normalize
  dW = dW/num_train + 2.0 * reg * W

  return loss, dW
