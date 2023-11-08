import random
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def reset_seed(number):
    """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
    """
    random.seed(number)
    torch.manual_seed(number)
    return


def get_toy_data(
    num_inputs=5,
    input_size=4,
    hidden_size=10,
    num_classes=3,
    dtype=torch.float32,
    device="cuda",
):
    """
    Get toy data for use when developing a two-layer-net.

    Inputs:
    - num_inputs: Integer N giving the data set size
    - input_size: Integer D giving the dimension of input data
    - hidden_size: Integer H giving the number of hidden units in the model
    - num_classes: Integer C giving the number of categories
    - dtype: torch datatype for all returned data
    - device: device on which the output tensors will reside

    Returns a tuple of:
    - toy_X: `dtype` tensor of shape (N, D) giving data points
    - toy_y: int64 tensor of shape (N,) giving labels, where each element is an
      integer in the range [0, C)
    - params: A dictionary of toy model parameters, with keys:
      - 'W1': `dtype` tensor of shape (D, H) giving first-layer weights
      - 'W2': `dtype` tensor of shape (H, C) giving second-layer weights
    """
    N = num_inputs
    D = input_size
    H = hidden_size
    C = num_classes

    # We set the random seed for repeatable experiments.
    reset_seed(0)

    # Generate some random parameters, storing them in a dict
    params = {}
    params["W1"] = 1e-4 * torch.randn(D, H, device=device, dtype=dtype)
    params["W2"] = 1e-4 * torch.randn(H, C, device=device, dtype=dtype)

    # Generate some random inputs and labels
    toy_X = 10.0 * torch.randn(N, D, device=device, dtype=dtype)
    toy_y = torch.tensor([0, 1, 2, 2, 1], device=device, dtype=torch.int64)

    return toy_X, toy_y, params


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-7):
    """
    Utility function to perform numeric gradient checking. We use the centered
    difference formula to compute a numeric derivative:

    f'(x) =~ (f(x + h) - f(x - h)) / (2h)

    Rather than computing a full numeric gradient, we sparsely sample a few
    dimensions along which to compute numeric derivatives.

    Inputs:
    - f: A function that inputs a torch tensor and returns a torch scalar
    - x: A torch tensor of the point at which to evaluate the numeric gradient
    - analytic_grad: A torch tensor giving the analytic gradient of f at x
    - num_checks: The number of dimensions along which to check
    - h: Step size for computing numeric derivatives
    """
    # fix random seed to 0
    reset_seed(0)
    for i in range(num_checks):

        ix = tuple([random.randrange(m) for m in x.shape])

        oldval = x[ix].item()
        x[ix] = oldval + h  # increment by h
        fxph = f(x).item()  # evaluate f(x + h)
        x[ix] = oldval - h  # increment by h
        fxmh = f(x).item()  # evaluate f(x - h)
        x[ix] = oldval  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error_top = abs(grad_numerical - grad_analytic)
        rel_error_bot = abs(grad_numerical) + abs(grad_analytic) + 1e-12
        rel_error = rel_error_top / rel_error_bot
        msg = "numerical: %f analytic: %f, relative error: %e"
        print(msg % (grad_numerical, grad_analytic, rel_error))


def compute_numeric_gradient(f, x, dLdf=None, h=1e-7):
    """
    Compute the numeric gradient of f at x using a finite differences
    approximation. We use the centered difference:

    df    f(x + h) - f(x - h)
    -- ~= -------------------
    dx           2 * h

    Function can also expand this easily to intermediate layers using the
    chain rule:

    dL   df   dL
    -- = -- * --
    dx   dx   df

    Inputs:
    - f: A function that inputs a torch tensor and returns a torch scalar
    - x: A torch tensor giving the point at which to compute the gradient
    - dLdf: optional upstream gradient for intermediate layers
    - h: epsilon used in the finite difference calculation
    Returns:
    - grad: A tensor of the same shape as x giving the gradient of f at x
    """
    flat_x = x.contiguous().flatten()
    grad = torch.zeros_like(x)
    flat_grad = grad.flatten()

    # Initialize upstream gradient to be ones if not provide
    if dLdf is None:
        y = f(x)
        dLdf = torch.ones_like(y)
    dLdf = dLdf.flatten()

    # iterate over all indexes in x
    for i in range(flat_x.shape[0]):
        oldval = flat_x[i].item()  # Store the original value
        flat_x[i] = oldval + h  # Increment by h
        fxph = f(x).flatten()  # Evaluate f(x + h)
        flat_x[i] = oldval - h  # Decrement by h
        fxmh = f(x).flatten()  # Evaluate f(x - h)
        flat_x[i] = oldval  # Restore original value

        # compute the partial derivative with centered formula
        dfdxi = (fxph - fxmh) / (2 * h)

        # use chain rule to compute dLdx
        flat_grad[i] = dLdf.dot(dfdxi).item()

    # Note that since flat_grad was only a reference to grad,
    # we can just return the object in the shape of x by returning grad
    return grad


def rel_error(x, y, eps=1e-10):
    """
    Compute the relative error between a pair of tensors x and y,
    which is defined as:

                            max_i |x_i - y_i]|
    rel_error(x, y) = -------------------------------
                      max_i |x_i| + max_i |y_i| + eps

    Inputs:
    - x, y: Tensors of the same shape
    - eps: Small positive constant for numeric stability

    Returns:
    - rel_error: Scalar giving the relative error between x and y
    """
    """ returns relative error between x and y """
    top = (x - y).abs().max().item()
    bot = (x.abs() + y.abs()).clamp(min=eps).max().item()
    return top / bot


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[torch.arange(N), y]
    margins = (x - correct_class_scores[:, None] + 1.0).clamp(min=0.)
    margins[torch.arange(N), y] = 0.
    loss = margins.sum() / N
    num_pos = (margins > 0).sum(dim=1)
    dx = torch.zeros_like(x)
    dx[margins > 0] = 1.
    dx[torch.arange(N), y] -= num_pos.to(dx.dtype)
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for
      the jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label
      for x[i] and 0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - x.max(dim=1, keepdim=True).values
    Z = shifted_logits.exp().sum(dim=1, keepdim=True)
    log_probs = shifted_logits - Z.log()
    probs = log_probs.exp()
    N = x.shape[0]
    loss = (-1.0 / N) * log_probs[torch.arange(N), y].sum()
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N
    return loss, dx


def load_data():
  train_dataset = datasets.MNIST(root='dataset/', train=True, 
                                transform=transforms.Compose([transforms.ToTensor()]), download=True)
  test_dataset = datasets.MNIST(root='dataset/', train=False, 
                                transform=transforms.Compose([transforms.ToTensor()]), download=True)
  train_loader = DataLoader(dataset=train_dataset, batch_size=60000, shuffle=True)
  test_loader = DataLoader(dataset=test_dataset, batch_size=10000)

  for data in train_loader:
    x_train, y_train = data

  for data in test_loader:
    x_test, y_test = data

  return x_train.reshape(x_train.shape[0], -1),y_train,x_test.reshape(x_test.shape[0], -1),y_test