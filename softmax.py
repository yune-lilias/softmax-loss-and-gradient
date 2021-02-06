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
  # here, it is easy to run into numeric instability. So use stable version   #
  # of softmax. Don't forget the regularization!                              #
  #############################################################################
  num_train = X.shape[0]
  #find c for stable version

  for i in xrange(num_train):
      scores = X[i].dot(W)
      typei = scores[y[i]]
      maxc = np.max(scores)
      fy = np.exp(typei - maxc)
      sumx = 0
      for j in scores:
        sumx = sumx + np.exp(j - maxc) 
      pi = fy/sumx
      loss = loss-np.log(pi)
    
      for k in xrange(W.shape[1]):
        pk = np.exp(scores[k]-maxc) / sumx
        if(y[i] == k):
          dW[:,k] = dW[:,k]+ (pk-1)*(X[i])
        else:
          dW[:,k] = dW[:,k]+ X[i]*(pk)
  dW = dW/num_train + 2*W
  reg2 = reg*np.square(W)
  loss = loss/num_train + reg2.sum()
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
  num_train = X.shape[0]
  scores = X.dot(W)
  maxc = np.max(scores,axis = 1)
  maxc = maxc.reshape(num_train,1)

  #meaning of so many reshape:
  # calculate between array of size(x,) and size(x,1) will not calculate between each element
  # but all the element with the second array with each element in the first

  temp = np.array(xrange(num_train)).reshape(num_train,1)*scores.shape[1] + y.reshape(num_train,1)
  sc1 = scores.reshape(scores.shape[1]*scores.shape[0],1)
  sc1 = sc1[temp].reshape(num_train,1)
  pi = np.exp(sc1 - maxc)/np.sum(np.exp(scores-maxc),axis = 1).reshape(num_train,1)

  p_all = np.exp(scores - maxc)/np.sum(np.exp(scores-maxc),axis = 1).reshape(num_train,1)

  reg = reg*np.square(W)
  loss = sum(-np.log(pi))/num_train + reg.sum()
  loss = loss[0]

  temp2 = np.zeros_like(p_all)
  temp2[np.arange(num_train), y] = 1
  dW = X.T.dot(p_all - temp2) / num_train + reg*2*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW

