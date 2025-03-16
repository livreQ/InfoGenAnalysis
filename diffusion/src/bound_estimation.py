'''
Date: 2024-05-18 01:19:12
LastEditors: qichen@cs.toronto.edu
LastEditTime: 2024-05-19 03:06:21
'''
#!/usr/bin/env python
# -*- coding=utf8 -*-

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KernelDensity
from numpy import linalg as LA
if torch.cuda.is_available():
    device = 'cuda'
    print('use gpu\n')
else:
    device = 'cpu'
    print('use cpu\n')

Log2PI = float(np.log(2 * np.pi))


def log_normal(x, mean, log_var, eps=0.00001):
    z = -0.5 * Log2PI
    return -(x - mean)**2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z


def sample_rademacher(shape):
    return (torch.rand(*shape).ge(0.5)).float() * 2 - 1


def sample_gaussian(shape):
    return torch.randn(*shape)


def sample_v(shape, vtype='rademacher'):
    if vtype == 'rademacher':
        return sample_rademacher(shape)
    elif vtype == 'normal' or vtype == 'gaussian':
        return sample_gaussian(shape)
    else:
        Exception(f'vtype {vtype} not supported')


class VariancePreservingSDE(torch.nn.Module):
    """
    Implementation of the variance preserving SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """

    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0, t_epsilon=0.001):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.t_epsilon = t_epsilon

    @property
    def logvar_mean_T(self):
        logvar = torch.zeros(1)
        mean = torch.zeros(1)
        return logvar, mean

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def mean_weight(self, t):
        return torch.exp(-0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min)

    def var(self, t):
        return 1. - torch.exp(-0.5 * t**2 * (self.beta_max - self.beta_min) - t * self.beta_min)

    def f(self, t, y):
        # f = alpha(t)y_t, alpha(t)= -1/2beta(t)
        return -0.5 * self.beta(t) * y

    def g(self, t, y):
        #\sqrt{beta_t} = g(t)
        beta_t = self.beta(t)
        return torch.ones_like(y) * beta_t**0.5

    def sample(self, t, y0, return_noise=False):
        """
        sample yt | y0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        mu = self.mean_weight(t) * y0
        std = self.var(t)**0.5
        epsilon = torch.randn_like(y0)
        yt = epsilon * std + mu
        if not return_noise:
            return yt
        else:
            return yt, epsilon, std, self.g(t, yt)

    def sample_debiasing_t(self, shape):
        """
        non-uniform sampling of t to debias the weight std^2/g^2
        the sampling distribution is proportional to g^2/std^2 for t >= t_epsilon
        for t < t_epsilon, it's truncated
        """
        raise NotImplementedError('See the official repository.')
        # return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, t_epsilon=self.t_epsilon, T=self.T)


class PluginReverseSDE(torch.nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    """

    def __init__(self, base_sde, drift_a, T, vtype='rademacher', debias=False):
        super().__init__()
        self.base_sde = base_sde
        self.a = drift_a
        self.T = T
        self.vtype = vtype
        self.debias = debias

    # Drift
    def mu(self, t, y, lmbd=0.):
        return (1. - 0.5 * lmbd) * self.base_sde.g(self.T-t, y) * self.a(y, self.T - t.squeeze()) - \
               self.base_sde.f(self.T - t, y)

    # Diffusion
    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd)**0.5 * self.base_sde.g(self.T - t, y)

    def score(self, t, y):
        return self.a(y, t) / self.base_sde.g(t, y)

    @torch.enable_grad()
    def dsm(self, x):
        """
        denoising score matching loss
        """
        #x--batch of train data
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([
                x.size(0),
            ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([
                x.size(0),
            ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
            #
        y, target, std, g = self.base_sde.sample(t_, x, return_noise=True)  #inf_sde
        a = self.a(y, t_.squeeze())
        #a -- g(t)s_theta(y, t)
        #         print(std, "std")

        return ((a * std / g + target)**2).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def elbo_random_t_slice(self, x):
        """
        estimating the ELBO of the plug-in reverse SDE by sampling t uniformly between [0, T], and by estimating
        div(mu) using the Hutchinson trace estimator
        """
        t_ = torch.rand([
            x.size(0),
        ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        qt = 1 / self.T
        y = self.base_sde.sample(t_, x).requires_grad_()

        a = self.a(y, t_.squeeze())
        mu = self.base_sde.g(t_, y) * a - self.base_sde.f(t_, y)

        v = sample_v(x.shape, vtype=self.vtype).to(y)

        Mu = -(torch.autograd.grad(mu, y, v, create_graph=self.training)[0] * v).view(x.size(0), -1).sum(1, keepdim=False) / qt

        Nu = -(a**2).view(x.size(0), -1).sum(1, keepdim=False) / 2 / qt
        yT = self.base_sde.sample(torch.ones_like(t_) * self.base_sde.T, x)
        lp = log_normal(yT, torch.zeros_like(yT), torch.zeros_like(yT)).view(x.size(0), -1).sum(1)

        return lp + Mu + Nu


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x


class MLP(nn.Module):

    def __init__(
            self,
            input_dim=2,
            index_dim=1,
            hidden_dim=128,
            act=Swish(),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim
        self.hidden_dim = hidden_dim
        self.act = act

        self.main = nn.Sequential(
            nn.Linear(input_dim + index_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, input, t):
        # init
        sz = input.size()
        input = input.view(-1, self.input_dim)
        t = t.view(-1, self.index_dim).float()

        # forward
        h = torch.cat([input, t], dim=1)  # concat
        output = self.main(h)  # forward
        return output.view(*sz)


def euler_maruyama_sampler(sde, x_0, num_steps, lmbd=0., keep_all_samples=True):
    """
    Euler Maruyama method with a step size delta
    """
    # init
    device = sde.T.device
    batch_size = x_0.size(0)
    ndim = x_0.dim() - 1
    #     print(sde.T.type)
    T_ = sde.T.cpu().item()
    delta = T_ / num_steps
    ts = torch.linspace(0, 1, num_steps + 1) * T_
    # num_stepms -- N
    # sample
    xs = []
    x_t = x_0.detach().clone().to(device)
    t = torch.zeros(batch_size, *([1] * ndim), device=device)
    max_L = 0
    mean_L = 0
    sum_beta = 0
    with torch.no_grad():
        for i in range(num_steps):
            t.fill_(ts[i].item())
            mu = sde.mu(t, x_t, lmbd=lmbd)
            sigma = sde.sigma(t, x_t, lmbd=lmbd)
            score = sde.score(t, x_t).to('cpu').numpy()
            score_norm = LA.norm(score, axis=1)
            sum_beta += sde.base_sde.beta(t).to('cpu').numpy()
            #             print("beta t", sde.base_sde.beta(t), "t", t)
            max_L = max(max_L, np.max(score_norm))
            mean_L = max(np.mean(score_norm), mean_L)
            x_t = x_t + delta * mu + delta**0.5 * sigma * torch.randn_like(x_t)  # one step update of Euler Maruyama method with a step size delta
            if keep_all_samples or i == num_steps - 1:
                #                 print("last time", i)
                xs.append(x_t.to('cpu').numpy())
            else:
                pass
        max_L = mean_L
    return xs, max_L, sum_beta


from sklearn.datasets import make_swiss_roll
from scipy.stats import multivariate_normal


class SwissRoll:
    """
    Swiss roll distribution sampler.
    noise control the amount of noise injected to make a thicker swiss roll
    """

    def sample(self, n, noise=0.5):
        if noise is None:
            noise = 0.5
        return torch.from_numpy(make_swiss_roll(n_samples=n, noise=noise)[0][:, [0, 2]].astype('float32') / 5.)


@torch.no_grad()
def cal_kl_datasets(pdata, qdata, bw=0.01):
    pdata = pdata  #.to('cpu').numpy()
    qdata = qdata  #.data.numpy()
    kde_p = KernelDensity(kernel='gaussian', bandwidth=bw).fit(pdata)
    logp = kde_p.score_samples(pdata)
    kde_q = KernelDensity(kernel='gaussian', bandwidth=bw).fit(qdata)
    logq = kde_q.score_samples(pdata)
    return logp.mean() - logq.mean()


@torch.no_grad()
def cal_kl_density_data(p_mean, p_var, qdata, bw=0.01, mc_samples=1000):
    # p is normal distribution
    pdata = np.random.multivariate_normal(mean=p_mean, cov=p_var, size=mc_samples)
    mvn = multivariate_normal(mean=p_mean, cov=p_var)
    #mvn2 = multivariate_normal(mean=[0,0], cov=[1,1])
    logp = mvn.logpdf(pdata)
    kde_q = KernelDensity(kernel='gaussian', bandwidth=bw).fit(qdata)
    logq = kde_q.score_samples(pdata)
    #logq = mvn2.logpdf(pdata)
    return logp.mean() - logq.mean()


@torch.no_grad()
def cal_kl_mvn(to, fr):
    """Calculate `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices"""
    import scipy
    m_to, S_to = to
    m_fr, S_fr = fr

    d = m_fr - m_to

    c, lower = scipy.linalg.cho_factor(S_fr)

    def solve(B):
        return scipy.linalg.cho_solve((c, lower), B)

    def logdet(S):
        return np.linalg.slogdet(S)[1]

    term1 = np.trace(solve(S_to))
    term2 = logdet(S_fr) - logdet(S_to)
    term3 = d.T @ solve(d)
    return (term1 + term2 + term3 - len(d)) / 2.


def gen_train_test_data(n_train=100, n_test=1000000):
    sampler = SwissRoll()
    x_train = sampler.sample(n_train)  #.data.numpy()
    x_test = sampler.sample(n_test)  #.data.numpy()
    return x_train, x_test


@torch.no_grad()
def evaluate(gen_sde, x_test):
    gen_sde.eval()
    num_samples = x_test.size(0)
    test_elbo = gen_sde.elbo_random_t_slice(x_test)
    gen_sde.train()
    return test_elbo.mean(), test_elbo.std() / num_samples**0.5


@torch.no_grad()
def evaluate_upper_bounds(gen_sde, x_train, sm_loss, R=0.01, num_steps=1000, mc_samples=1000):
    # bound value given train data size m, score estimation iter, diffusion time T, subgaussian variable R
    T = gen_sde.T
    #     print(T.type)
    varT = gen_sde.base_sde.var(T)
    varT = varT.to('cpu').numpy()[0]
    m = x_train.shape[0]
    dim = x_train.shape[1]
    qdata = []
    #     path_x = []
    #     path_all = []
    xs, max_L, sum_beta = euler_maruyama_sampler(gen_sde, x_train, num_steps, lmbd=0., keep_all_samples=False)
    print("max norm", max_L)
    for x in x_train:
        muT = gen_sde.base_sde.mean_weight(T) * x.to(device)  # mean of E_T(X_i)
        qdata.append(muT.to('cpu').numpy())

    avg_term1 = 0
    avg_term2 = 0
    avg_term3 = 0
    bound = 0
    print(x_train.shape)
    print(gen_sde.base_sde.mean_weight(T), "mean weight")
    print(varT, "varT")
    #print(qdata, "sample means")
    #     print(sum_beta, "sum_beta")

    delta = np.diag(np.ones(dim))
    for i in range(m):
        avg_term1 += -cal_kl_density_data(qdata[i], (varT**0.5) * delta, qdata, bw=(varT**0.25), mc_samples=mc_samples)  # KL(E_T(X_i)||E_T#\hat{P}_X)
        avg_term2 += np.sqrt(cal_kl_mvn([qdata[i], varT * delta], [np.zeros(dim), delta]))  # KL(E_T(X_i)||pi))


#         avg_term3 += np.sqrt(cal_kl_datasets(path_x[i], path_all, bw=0.01))

    avg_term1 /= m
    avg_term2 /= m
    avg_term3 = np.sqrt(T.to('cpu').numpy() * max_L**2 / (2 * m * num_steps) * sum_beta[0][0])
    print(avg_term1, avg_term2, avg_term3)
    bound = avg_term1 + np.sqrt(2) * R * avg_term2 + np.sqrt(2) * R * avg_term3 + sm_loss
    return bound


def cal_generation_error(gen_sde, x_test, mc_samples=10000):
    if torch.cuda.is_available():
        device = 'cuda'
        print('use gpu\n')
    else:
        device = 'cpu'
        print('use cpu\n')
    x_0 = torch.randn(mc_samples, 2, device=device)  # init from prior
    xs, max_L, sum_beta = euler_maruyama_sampler(gen_sde, x_0, num_steps, lmbd=0)  # sample
    # print(x_0.shape, "x_0", np.shape(xs), "xs", np.shape(x_test), "x_test")
    x_generated_lastT = xs[-1][:mc_samples]
    kl_div = cal_kl_datasets(x_test, x_generated_lastT, 0.1)
    return kl_div, x_generated_lastT

def train_sde(X_train, X_test, T0 = 1, batch_size = 256, iterations = 10000,\
              lr = 0.001, print_every = 5000, vtype = 'rademacher', \
              num_steps=10, mc_samples=100, R= 0.1):
    # init device
    if torch.cuda.is_available():
        device = 'cuda'
        print('use gpu\n')
    else:
        device = 'cpu'
        print('use cpu\n')
    # init models
    drift_q = MLP(input_dim=2, index_dim=1, hidden_dim=128).to(device)
    T = torch.nn.Parameter(torch.FloatTensor([T0]), requires_grad=False)
    inf_sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=T).to(device)
    gen_sde = PluginReverseSDE(inf_sde, drift_q, T, vtype=vtype, debias=False).to(device)
    upper_bounds = []
    test_kl = []
    sm_losses = []
    x_generated = []
    # init optimizer
    optim = torch.optim.Adam(gen_sde.parameters(), lr=lr)

    # train
    start_time = time.time()
    # iteration for trainning of theta
    for i in range(iterations):
        optim.zero_grad()  # init optimizer
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        score = []
        for start in range(0, X_train.shape[0], batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            #           x = sampler.sample(batch_size).to(device) # sample data
            x = X_batch.to(device)
            loss = gen_sde.dsm(x).mean()  # forward and compute loss
            loss.backward()  # backward
            optim.step()  # update
            score.append(loss.item())
        avg_score = np.mean(score)

        # print
        if (i + 1) % print_every == 0:
            # elbo
            #             elbo, elbo_std = evaluate(gen_sde, X_train)
            bound = evaluate_upper_bounds(gen_sde, X_train, avg_score, R, num_steps, mc_samples)
            upper_bounds.append(bound)
            sm_losses.append(avg_score)
            print(X_test.shape, "x test")
            elapsed = time.time() - start_time
            print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f}'.format(i + 1, elapsed * 1000 / print_every, loss.item()))
            kl_div, x_generated_lastT = cal_generation_error(gen_sde, X_test, mc_samples=mc_samples)
            print(kl_div)
            test_kl.append(kl_div)
            x_generated.append(x_generated_lastT)
            print('| iter {:6d} | {:5.2f} ms/step | kl_div {} | bound {}'.format(i + 1, elapsed * 1000 / print_every, kl_div, bound))
            start_time = time.time()
    return gen_sde, upper_bounds, test_kl, sm_losses, x_generated


if __name__ == '__main__':
    mc_samples = 1000
    num_steps = 1000
    n_train = 10  #m
    T = 1  #
    n_test = 1000

    import random
    sm_training_iter = 10000
    rep = 3
    for n_train in [10, 100, 200, 600, 800, 1000, 2000]:
        for i in range(rep):
            random.seed(i + 20)
            prefix = "m_" + str(n_train) + "_rep_" + str(i) + "_T_" + str(T) + "_iter_" + str(sm_training_iter)
            x_train, x_test = gen_train_test_data(n_train=n_train, n_test=n_test)
            x_test = x_test.to('cpu').numpy()
            gen_sde, upper_bounds, test_kl, sm_losses, x_generated = train_sde(x_train, \
                                                                    x_test, \
                                                                    T0 = T, \
                                                                    iterations=sm_training_iter, \
                                                                    print_every = 1000, \
                                                                    num_steps=num_steps, \
                                                                    mc_samples=mc_samples,\
                                                                    R = 5)
            with open("./output/results/toy/" + prefix + '.npy', 'wb') as f:
                np.save(f, upper_bounds)
                np.save(f, test_kl)
                np.save(f, sm_losses)
                np.save(f, x_generated)
    '''
    n_train = 200 
    for a in range(2, 20, 2):
        T = a/10.0
        for i in range(rep):
            random.seed(i + 20)
            prefix = "m_" + str(n_train) + "_rep_" + str(i) + "_T_" + str(T) + "_iter_" + str(sm_training_iter)
            x_train, x_test = gen_train_test_data(n_train=n_train, n_test=n_test)
            x_test = x_test.to('cpu').numpy()
            gen_sde, upper_bounds, test_kl, sm_losses, x_generated = train_sde(x_train, \
                                                                    x_test, \
                                                                    T0 = T, \
                                                                    iterations=sm_training_iter, \
                                                                    print_every = 10000, \
                                                                    num_steps=num_steps, \
                                                                    mc_samples=mc_samples,\
                                                                    batch_size=n_train,
                                                                    R = 5)
            with open("./all_result/" + prefix + '.npy', 'wb') as f:
                #np.save(f, gen_sde)
                np.save(f, upper_bounds)
                np.save(f, test_kl)
                np.save(f, sm_losses)
                np.save(f, x_generated)'''
