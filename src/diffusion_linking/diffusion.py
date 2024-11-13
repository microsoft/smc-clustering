# Licensed under the MIT license.

import torch

from .model import SetFormer
from .schedule import LinearSchedule


class VariationalDiffusion(torch.nn.Module):
    """
    Variational diffusion following https://arxiv.org/pdf/2107.00630.pdf

    In this class, we instantiate a SetFormer model, and pass it a concatenation of
    the noisy state z, with the number of (not-masked out) elements of the set, and the diffusion time.
    The model is trained to predict the noise that was added to z.

    We use a "schedule" to control the variance of the noise added to z.
    We require that the schedule is a monotone function of time, that it is positive and
      f(0) = 0
      f(1) = 1
    (or approximately so, in the case of the linear noise schedule)

    The loss function involves the gradient of the logit of the schedule!

    If f(t) = sigmoid(gamma(t)), then the loss involves the gradient of gamma(t) with respect to t.
    """

    def __init__(self, dim, depth, schedule=None):
        super().__init__()
        self.dim = dim

        if schedule is None:
            schedule = LinearSchedule()
        self.schedule = schedule

        self.net = SetFormer(dim, depth)

    def forward(self, x, masks):
        batch_size, seq_len, _ = x.shape

        # sample time points across batch
        # ti = mod(u0 + i/batch_size, 1)
        u0 = torch.rand(1, dtype=x.dtype, device=x.device)
        t = torch.arange(batch_size, dtype=x.dtype, device=x.device) / batch_size + u0
        t %= 1
        t = t[:, None, None]  # bcast over seq, dim

        # set up noise schedule
        sigma2 = self.schedule(t)
        sigma = torch.sqrt(sigma2)
        alpha = torch.sqrt(1 - sigma2)

        # sample noise and add to data
        eps = torch.randn_like(x)
        z = alpha * x + sigma * eps

        # pass z, set_size and t through the model, concatenated
        set_size = masks.sum(dim=1)[:, None, None]
        model_input = torch.cat([z, set_size.repeat(1, seq_len, 1), t.repeat(1, seq_len, 1)], dim=-1)
        eps_hat = self.net(model_input, masks)

        # compute the L_inf loss
        gamma_grad = self.schedule.gamma_grad(t)
        loss = 0.5 * torch.sum(gamma_grad * masks.to(x.dtype)[:, :, None] * (eps_hat - eps) ** 2, dim=[1, 2])
        loss = loss.mean()  # average over batch

        return loss

    @torch.no_grad()
    def smc_step(self, s, t, z_s, log_weights, log_prob):
        # compute mean and variance of q(z_t | z_s)
        mu_q_ts, sigma2_q_ts = self.moments_q_ts(z_s, t, s)

        # sample z_t | z_s
        eps = torch.randn(*z_s.shape, device=z_s.device)
        z_t = mu_q_ts + torch.sqrt(sigma2_q_ts) * eps

        # compute mean and variance of p(z_s | z_t) (includes network call)
        mu_p_st, sigma2_p_st = self.moments_p_st(z_t, s, t)

        # compute log( p(z_s | z_t) / q(z_t | z_s) )
        log_p = -0.5 * torch.sum(torch.log(sigma2_p_st) + (z_s - mu_p_st) ** 2 / sigma2_p_st, dim=[1, 2])
        log_q = -0.5 * torch.sum(torch.log(sigma2_q_ts) + (z_t - mu_q_ts) ** 2 / sigma2_q_ts, dim=[1, 2])
        log_ratios = log_p - log_q

        # update weights, normalizer
        log_weights += log_ratios
        log_normalizer = torch.logsumexp(log_weights, dim=0)
        log_weights -= log_normalizer
        log_prob += log_normalizer

        return z_t, log_weights, log_prob

    @torch.no_grad()
    def logp_smc(self, x, num_particles, num_time_steps, resample_thresh=0.25):
        """
        This runs a loop forwards in time, starting at x=z_0.
        At each smc step we sample from q(z_t | z_s) (where s < t),
        and weight the samples by log (p(z_t | z_s) / q(z_t | z_s))
        the log weights are then normalized to sum to 1, and the product
        of those weights is the particle estimate of p(z_0).
        """
        z_s = x.repeat(num_particles, 1, 1)

        log_prob = 0.0
        log_weights = torch.ones(num_particles, device=x.device) * torch.log(torch.tensor(1.0 / num_particles))
        for i in torch.arange(0, num_time_steps, device=x.device):
            # t is the "next" time step, s is the "current" time step
            s = i / num_time_steps
            t = (i + 1) / num_time_steps

            z, log_weights, log_prob = self.smc_step(s, t, z_s, log_weights, log_prob)

            # resample particles if degenerate
            n_eff = 1.0 / torch.sum(torch.square(torch.exp(log_weights)))
            # print(f"iteration {i}, logp = {log_prob.item():.2f}, n_eff = {n_eff.item():.2f}")
            if n_eff < resample_thresh * num_particles:
                z = z[torch.multinomial(torch.exp(log_weights), num_particles)]
                log_weights = torch.ones(num_particles, device=x.device) * torch.log(torch.tensor(1.0 / num_particles))

        return log_prob, z, log_weights

    @torch.no_grad()
    def moments_q_ts(self, z_s, t, s):
        """
        Compute the mean and variance of q(z_t | z_s).

        See eqs 20-22 of the paper (https://arxiv.org/pdf/2107.00630.pdf).

        note s < t
        """
        sigma2_t = self.schedule(t)
        alpha2_t = 1 - sigma2_t
        sigma2_s = self.schedule(s)
        alpha2_s = 1 - sigma2_s
        alpha2_ts = alpha2_t / alpha2_s
        mu = z_s * torch.sqrt(alpha2_ts)
        sigma2_ts = sigma2_t - alpha2_ts * sigma2_s

        return mu, sigma2_ts

    @torch.no_grad()
    def moments_p_st(self, z, s, t, masks=None):
        """
        Compute the mean and variance of p(z_s | z_t)
            according to eq. 34 of the paper (https://arxiv.org/pdf/2107.00630.pdf).

        note s < t.
        """
        if masks is None:
            masks = torch.ones_like(z[:, :, 0], dtype=torch.bool)
        batch_size, seq_len, _ = z.shape
        device = z.device
        set_size = masks.sum(dim=1)[:, None, None]
        model_input = torch.cat(
            [z, set_size.repeat(1, seq_len, 1), t * torch.ones(batch_size, seq_len, 1, device=device)], dim=-1
        )
        eps_hat = self.net.eval()(model_input)

        gamma_s = self.schedule.gamma(s)
        gamma_t = self.schedule.gamma(t)
        sigma2_t = self.schedule(t)
        sigma_t = torch.sqrt(sigma2_t)
        alpha2_t = 1 - sigma2_t
        sigma2_s = self.schedule(s)
        alpha2_s = 1 - sigma2_s
        c = -torch.expm1(gamma_s - gamma_t)
        mu = torch.sqrt(alpha2_s / alpha2_t) * (z - sigma_t * c * eps_hat)
        sigma2 = sigma2_s * c
        return mu, sigma2

    @torch.no_grad()
    def generate(self, num_samples: int, seq_len: int, num_time_steps: int, device, masks=None):
        # generate samples from the model

        if masks is None:
            masks = torch.ones(num_samples, seq_len, dtype=torch.bool, device=device)

        z = torch.randn(num_samples, seq_len, self.dim, device=device)

        for i in torch.arange(0, num_time_steps, device=device):
            # in each step, we will sample from p(z_s | z_t)
            # where t starts from 1 and goes down to 1/num_time_steps
            # and s starts from 1 - 1/num_time_steps and goes down to 0
            # s < t.
            t = (num_time_steps - i) / num_time_steps
            s = (num_time_steps - i - 1) / num_time_steps

            mu, sigma2 = self.moments_p_st(z, s, t, masks)
            z = mu + torch.sqrt(sigma2) * torch.randn_like(z)

        return z

    def score_fn(self, z, masks, t):
        """
        Compute the score function, which is quite close to the noise prediction function.
        See https://arxiv.org/pdf/2107.00630.pdf eqs 29-31.
        """
        batch_size, seq_len, _ = z.shape
        # concatenate z, set_size and t
        set_size = masks.sum(dim=1)[:, None, None]
        model_input = torch.cat(
            [z, set_size.repeat(1, seq_len, 1), t * torch.ones(batch_size, seq_len, 1, device=z.device)], dim=-1
        )
        eta_hat = self.net.eval()(model_input, masks)
        sigma2 = self.schedule(t)
        sigma = torch.sqrt(sigma2)
        return -eta_hat / sigma

    def prob_flow_grad_fn(self, z, masks, t):
        # taken from https://arxiv.org/pdf/2210.05475.pdf
        # Eq 18 and surrounding notes
        s = self.score_fn(z, masks, t)
        beta = self.schedule(t) * self.schedule.gamma_grad(t)
        return -0.5 * beta * (z + s)

    def generate_ode(self, num_samples, seq_len, num_time_steps, device, masks=None, z1=None):
        """
        Using the probability flow ODE to generate samples.

        The ode is:

            dz = -beta * (z + score(z)) dt

        Note that we run the ode _backwards_ in time, from t=1 to t=0.
        This just uses a simple Euler expansion for now.
        """
        if masks is None:
            masks = torch.ones(num_samples, seq_len, dtype=torch.bool, device=device)
        if z1 is None:
            z1 = torch.randn(num_samples, seq_len, self.dim, device=device)

        z = z1
        dt = 1 / num_time_steps
        for i in torch.arange(0, num_time_steps, device=device):
            t = torch.tensor(1 - (i * dt))
            z = z - dt * self.prob_flow_grad_fn(z, masks, t)
        return z

    @torch.no_grad()
    def log_prob_ode(self, x, masks=None, num_time_steps=100):
        """
        Compute the log probability of x under the model, using the probability flow ODE.
        """
        if masks is None:
            masks = torch.ones_like(x[:, :, 0], dtype=torch.bool)

        dt = 1 / num_time_steps
        log_prob = 0.0
        for i in torch.arange(0, num_time_steps, device=x.device):
            # use vector-Jacobian product and Skilling-Hutchinson estimator
            # see https://openreview.net/pdf?id=PxTIG12RRHS eq 39 - 40
            t = torch.tensor(i * dt)
            eps = torch.randn_like(x)
            f = lambda z: self.prob_flow_grad_fn(z, masks, t)
            grad, vjp = torch.autograd.functional.vjp(f, x, v=eps, create_graph=False, strict=True)
            x = x + dt * grad

            log_prob += torch.sum(eps * vjp * masks.to(eps.dtype)[:, :, None], dim=[1, 2]) * dt

        log_p_xT = -0.5 * (x**2 + 2 * torch.log(torch.tensor(2 * 3.141592653589793)))
        log_prob += torch.sum(torch.where(masks, log_p_xT.sum(dim=-1), 0), dim=-1)
        return log_prob, x
