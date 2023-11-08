import torch
from utils import svm_loss, softmax_loss

def hello_fully_connected_networks():
    print('Hello from fully_connected_networks.py!')


class Linear(object):

    @staticmethod
    def forward(x, w):
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w)
        """
        out = None
        out = x.view(x.shape[0],-1).mm(w)
        cache = (x, w)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        """
        x, w = cache
        dx, dw = None, None
        dx = dout.mm(w.t()).view(x.shape)
        dw = x.view(x.shape[0],-1).t().mm(dout)
        return dx, dw


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = None
        ###################################################
        # TODO: Implement the ReLU forward pass.          #
        # You should not change the input tensor x with an#
        # in-place operation. Try to clone it first.      #
        ###################################################
        # Replace "pass" statement with your code
        out = x.clone()  # Make a copy of the input tensor
        out[out < 0] = 0  # Apply ReLU activation element-wise

        ###################################################
        #                 END OF YOUR CODE                #
        ###################################################
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        #####################################################
        # TODO: Implement the ReLU backward pass.           #
        # You should not change the input tensor dout with  #
        # an in-place operation. Try to clone it first.     #
        #####################################################
        # Replace "pass" statement with your code
        dx = dout.clone()
        dx[x < 0] = 0
        #####################################################
        #                  END OF YOUR CODE                 #
        #####################################################
        return dx


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg

        # Initialize
        self.params['W1'] = torch.zeros(input_dim, hidden_dim, dtype=dtype,device = device)
        self.params['W1'] += weight_scale*torch.randn(input_dim, hidden_dim, dtype=dtype,device= device)
        self.params['W2'] = torch.zeros(hidden_dim, num_classes, dtype=dtype,device= device)
        self.params['W2'] += weight_scale*torch.randn(hidden_dim, num_classes, dtype=dtype,device= device)

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'params': self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the
          label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Tensor of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i]
          and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        scores = None
        #############################################################
        # TODO: Implement the forward pass for the two-layer net,   #
        # computing the class scores for X and storing them in the  #
        # scores variable.                                          #
        #############################################################
        # Replace "pass" statement with your code
        # Forward pass: Linear -> ReLU -> Linear
        out1, cache1 = Linear.forward(X, self.params['W1'])
        out2, cache2 = ReLU.forward(out1)
        scores, cache3 = Linear.forward(out2, self.params['W2'])
        ##############################################################
        #                     END OF YOUR CODE                       #
        ##############################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}

        # compute the loss and gradient for softmax classification
        loss, dout = softmax_loss(scores, y)
        ###################################################################
        # TODO: Implement the backward pass for the two-layer net.        #
        # The upstream derivatives "dout" have been given.                #
        # You just need to compute gradients of Linear and ReLU layer     #
        ###################################################################
        # Replace "pass" statement with your code
        dout2, grads['W2'] = Linear.backward(dout, cache3)
        dout1 = ReLU.backward(dout2, cache2)
        dx, grads['W1'] = Linear.backward(dout1, cache1)


        loss += 0.5 * self.reg * (torch.sum(self.params['W1'] ** 2) + torch.sum(self.params['W2'] ** 2))

        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        ###################################################################
        #                     END OF YOUR CODE                            #
        ###################################################################

        return loss, grads
        

def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config