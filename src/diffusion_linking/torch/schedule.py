# Licensed under the MIT license.

import torch


class MonotoneNetSchedule(torch.nn.Module):
    """
    This monotone neural net is based on a simple sigmoid activation function.

    The form is
        f(x) = sigmoid((x - b) @ W1) @ W2

    where W1 and W2 are positive matrices. We enforce this by using the softplus
    activation function on W1, and softmax on W2.

    We also define a "normalized function" which is f(x) normalized to be between 0 and 1.

        g(x) = (f(x) - f(0)) / (f(1) - f(0))

    This is the main function, used for the noise variance in the diffusion model.

    We also define a "gamma" function which is the logit of the normalized function.

        gamma(x) = log(g(x) / (1-g(x)))

    so that

        g(x) = sigmoid(gamma(x))

    I've tried to implement these in a numerically stable way.

    Finally, we define the gradient of gamma with respect to x, which is required for training a diffusion model.
    """

    def __init__(self, hidden_size, epsilon=1e-5):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.hidden_size = hidden_size
        self.W1 = torch.nn.Parameter(torch.randn(self.input_dim, self.hidden_size) + 50)
        self.b1 = torch.nn.Parameter(torch.rand(self.hidden_size))
        self.W2 = torch.nn.Parameter(torch.randn(self.hidden_size, self.output_dim))
        self.epsilon = epsilon

    def forward(self, x):
        return torch.sigmoid(self.gamma(x))

    def compute(self, x, derivative=False):
        """
        Compute the function f(x), or f'(x) if derivative=True.
        """
        # make sure x is 2d
        orig_shape = x.shape
        x = x.reshape(-1, 1)

        # make weights positive to ensure monotonicity
        W1 = torch.nn.Softplus()(self.W1)
        W2 = torch.nn.Softmax(dim=0)(self.W2)

        # forward pass
        tmp = torch.sigmoid((x - self.b1) * W1)
        if derivative:
            x = (tmp * (1 - tmp)) @ (W2 * W1.T)
        else:
            x = tmp @ W2

        # reshape as needed
        x = x.reshape(orig_shape)

        return x

    def normalized_function(self, t):
        """
        Construct a function from the neural net s.t. f(0) = 0 and f(1) = 1.
        Note that this never gets called, but is here for reference: it should be equivalent to self.forward.
        """
        f_t = self.compute(t)
        f_0 = self.compute(torch.zeros_like(t))
        f_1 = self.compute(torch.ones_like(t))
        g_t = (f_t - f_0) / (f_1 - f_0)
        return g_t * (1 - 2 * self.epsilon) + self.epsilon

    def gamma(self, t):
        """
        Construct a function gamma from the neural net such that
          gamma = logit(normalized_function(t)).

        This means that gamma(0) = -inf and gamma(1) = inf.

        We use a small epsilon to avoid log(0).
        """
        f_t = self.compute(t)
        f_0 = self.compute(torch.zeros_like(t))
        f_1 = self.compute(torch.ones_like(t))

        f_t_0 = (f_t - f_0) * (1 - 2 * self.epsilon)
        return torch.log(f_t_0 + self.epsilon * f_1) - torch.log(f_1 * (1 - self.epsilon) - f_t_0)

    def gamma_grad(self, t):
        """
        Compute the gradient of gamma with respect to t.
        """
        f_t = self.compute(t)
        f_0 = self.compute(torch.zeros_like(t))
        f_1 = self.compute(torch.ones_like(t))
        f_grad = self.compute(t, derivative=True)
        f_t_0 = (f_t - f_0) * (1 - 2 * self.epsilon)
        gamma_grad = f_grad * (1 - 2 * self.epsilon) / (self.epsilon * f_1 + f_t_0) / (f_1 * (1 - self.epsilon) - f_t_0)
        return gamma_grad


class LinearSchedule(torch.nn.Module):
    """
    If you want something simpler than the Monotone net above, try out this linear scheduler.

    we use gamma(x) = a + (b-a) * x

    so that gamma(0) = a and gamma(1) = b

    a should be a negative number, b should be positive. The resulting diffusion model is only
    "correct" as a -> -inf and b -> inf, but there are no parameters to tune.
    """

    def __init__(self, gamma_0=-10.0, gamma_1=10.0):
        super(LinearSchedule, self).__init__()
        self.gamma_0 = torch.tensor(gamma_0)
        self.gamma_1 = torch.tensor(gamma_1)

    def gamma(self, t):
        return self.gamma_0 + (self.gamma_1 - self.gamma_0) * t

    def gamma_grad(self, t):
        return (self.gamma_1 - self.gamma_0) * torch.ones_like(t)

    def forward(self, t):
        return torch.sigmoid(self.gamma(t))
