#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# File Name: img_bound_estimation.py
# Description:
"""

import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.neighbors import KernelDensity
from numpy import linalg as LA
from models.sdes import VariancePreservingSDE, PluginReverseSDE
from models.unet import UNet
from scipy.stats import multivariate_normal
from utils.helpers import logging
# from src.utils.plotting import get_grid
from tensorboardX import SummaryWriter
import argparse
import os
import json
from flows.elemwise import LogitTransform

if torch.cuda.is_available():
    device = 'cuda'
    print('use gpu\n')
else:
    device = 'mps'
    print('use mps\n')


def get_grid(sde, input_channels, input_height, n=4, num_steps=20, transform=None, mean=0, std=1, clip=True):
    """
        sample n^2 images 
    """
    num_samples = n**2
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, input_channels, input_height, input_height).to(sde.T)
    #noise
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0)
            sigma = sde.sigma(ones * ts[i], y0)
            y0 = y0 + delta * mu + delta**0.5 * sigma * torch.randn_like(y0)

    if transform is not None:
        y0 = transform(y0)

    if clip:
        y0 = torch.clip(y0, 0, 1)

    # ou_sde.sample(ou_sde.T*50, x)
    y0 = y0.view(n, n, input_channels, input_height, input_height).permute(2, 0, 3, 1, 4).contiguous().view(input_channels, n * input_height, n * input_height)

    y0 = y0.data.cpu().numpy()
    return y0


def euler_maruyama_sampler(sde, x_0, num_steps, lmbd=0., keep_all_samples=True, transform=None):
    """
    Euler Maruyama method with a step size delta
    """
    # init
    device = sde.T.device
    batch_size = x_0.size(0)
    ndim = x_0.dim() - 1
    print("x0 shape:", x_0.shape)
    T_ = sde.T.cpu().item()
    delta = T_ / num_steps
    ts = torch.linspace(0, 1, num_steps + 1) * T_
    ones = torch.ones(batch_size, 1, 1, 1).to(x_0)
    # num_stepms -- N
    # sample
    xs = []
    x_t = x_0.detach().clone().to(device)

    #t = torch.zeros(batch_size, *([1]*ndim), device=device)
    max_L = 0
    mean_L = 0
    sum_beta = 0
    with torch.no_grad():
        for i in range(num_steps):
            #t.fill_(ts[i].item())# fill ts[i] to t
            #print("ts:", ts, "ts[i]:",ts[i], "ones shape:", ones.shape)
            t = ones * ts[i]
            mu = sde.mu(t, x_t, lmbd=lmbd)
            #print("mu shape:", mu.shape)
            sigma = sde.sigma(t, x_t, lmbd=lmbd)

            score = sde.score(t, x_t).to('cpu').numpy()
            score_norm = LA.norm(score, axis=1)
            sum_beta += sde.base_sde.beta(t).to('cpu').numpy()
            #             print("beta t", sde.base_sde.beta(t), "t", t)
            max_L = max(max_L, np.max(score_norm))
            mean_L = max(np.mean(score_norm), mean_L)
            x_t = x_t + delta * mu + delta**0.5 * sigma * torch.randn_like(x_t)  # one step update of Euler Maruyama method with a step size delta
            if transform is not None:
                x_t = transform(x_t)
            if keep_all_samples or i == num_steps - 1:
                #                 print("last time", i)
                xs.append(x_t)
            else:
                pass
        max_L = mean_L
    return xs, max_L, sum_beta


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


def gen_train_test_data(dataset='mnist', datapath='./', train_batch_size=64, test_batch_size=64, n_train=1000, n_test=1000):
    dataroot = datapath + "/" + dataset
    if dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root=dataroot, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=dataroot, train=False, download=True, transform=transform)

    elif dataset == 'cifar10':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform)
    else:
        raise NotImplementedError

    sub_trainset_indices = torch.randperm(len(trainset))[:n_train]
    sub_testset_indices = torch.randperm(len(testset))[:n_test]
    sub_trainset = Subset(trainset, sub_trainset_indices)
    sub_testset = Subset(testset, sub_testset_indices)

    print("train data len, shape:", len(sub_trainset), sub_trainset)
    print("test data len, shape:", len(sub_testset), sub_testset)
    trainloader = DataLoader(sub_trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(sub_testset, batch_size=test_batch_size, shuffle=True, num_workers=2)
    return trainloader, testloader, transform, sub_trainset, sub_testset


# @torch.no_grad()
# def evaluate(gen_sde, x_test):
#     gen_sde.eval()
#     num_samples = x_test.size(0)
#     test_elbo = gen_sde.elbo_random_t_slice(x_test)
#     gen_sde.train()
#     return test_elbo.mean(), test_elbo.std() / num_samples ** 0.5


def get_score_model(dataset='mnist'):

    if dataset == 'mnist':
        input_channels = 1
        input_height = 28
        dimx = input_channels * input_height**2
        drift_q = UNet(
            input_channels=input_channels,
            input_height=input_height,
            ch=32,
            ch_mult=(1, 2, 2),
            num_res_blocks=2,
            attn_resolutions=(16, ),
            resamp_with_conv=True,
        )

    elif dataset == 'cifar10':
        input_channels = 3
        input_height = 32
        dimx = input_channels * input_height**2
        drift_q = UNet(
            input_channels=input_channels,
            input_height=input_height,
            ch=128,
            ch_mult=(1, 2, 2, 2),
            num_res_blocks=2,
            attn_resolutions=(16, ),
            resamp_with_conv=True,
        )
    else:
        raise NotImplementedError
    return drift_q, dimx, input_channels, input_height


@torch.no_grad()
def evaluate(gen_sde, testloader, real, logit, dimx, n_test):
    test_bpd = list()
    gen_sde.eval()
    for x_test, _ in testloader:
        x_test = x_test.to(device)
        x_test = x_test * 255 / 256 + torch.rand_like(x_test) / 256
        if real:
            x_test, ldj = logit.forward_transform(x_test, 0)
            elbo_test = gen_sde.elbo_random_t_slice(x_test)
            elbo_test += ldj
        else:
            elbo_test = gen_sde.elbo_random_t_slice(x_test)

        test_bpd.extend(-(elbo_test.data.cpu().numpy() / dimx) / np.log(2) + 8)
    gen_sde.train()
    test_bpd = np.array(test_bpd)
    print("testloader size, test data size:", len(testloader), n_test)
    return test_bpd.mean(), test_bpd.std() / (n_test)**0.5


@torch.no_grad()
def evaluate_upper_bounds(gen_sde, logit, x_train, n_train, sm_loss, R=0.01, num_steps=20, mc_samples=1000, real=True):
    # bound value given train data size m, score estimation iter, diffusion time T, subgaussian variable R
    T = gen_sde.T
    print(T.type)
    varT = gen_sde.base_sde.var(T)
    varT = varT.to('cpu').numpy()[0]
    print("number of batches:", len(x_train))
    print("number of images:", n_train)
    m = n_train
    max_L = 0
    #xs, max_L, sum_beta = euler_maruyama_sampler(gen_sde, x_train, num_steps, lmbd=0., keep_all_samples=False)
    print("max norm", max_L)
    flag = True
    for x, _ in x_train:
        x = x.to(device)
        x = x * 255 / 256 + torch.rand_like(x) / 256
        if real:
            x, _ = logit.forward_transform(x, 0)
        print("batch x shape:", x.shape)
        muT = gen_sde.base_sde.mean_weight(T) * x.to(device)  # mean of E_T(X_i)

        if flag:
            qdata = muT
            flag = False
            _, max_L, sum_beta = euler_maruyama_sampler(gen_sde, x, num_steps, lmbd=0., keep_all_samples=False)
        else:
            #qdata.append(muT.to('cpu').numpy())
            qdata = torch.cat((qdata, muT), 0)
    qdata = qdata.view(m, -1)
    dim = qdata.shape[1]
    assert qdata.shape[0] == m
    qdata = qdata.to('cpu').numpy()
    avg_term1 = 0
    avg_term2 = 0
    avg_term3 = 0
    bound = 0
    print(gen_sde.base_sde.mean_weight(T), "mean weight")
    print(varT, "varT")
    #print(qdata, "sample means")
    #     print(sum_beta, "sum_beta")

    delta = np.diag(np.ones(dim))
    for i in range(m):
        #print(i, "start cal data kl")
        avg_term1 += -cal_kl_density_data(qdata[i], (varT**0.5) * delta, qdata, bw=(varT**0.25), mc_samples=mc_samples)  # KL(E_T(X_i)||E_T#\hat{P}_X)
        #print(i, "start cal mvn kl")
        avg_term2 += np.sqrt(cal_kl_mvn([qdata[i], varT * delta], [np.zeros(dim), delta]))  # KL(E_T(X_i)||pi))


#         avg_term3 += np.sqrt(cal_kl_datasets(path_x[i], path_all, bw=0.01))

    avg_term1 /= m
    avg_term2 /= m
    avg_term3 = np.sqrt(T.to('cpu').numpy() * max_L**2 / (2 * m * num_steps) * sum_beta[0][0])
    print(avg_term1, avg_term2, avg_term3)
    bound = avg_term1 + np.sqrt(2) * R * avg_term2 + np.sqrt(2) * R * avg_term3 + sm_loss
    return bound


def cal_generation_error(gen_sde, x_test, mc_samples=10000, num_steps=1000, input_channels=1, input_height=28, transform=None):
    #x_0 = torch.randn(mc_samples, 2, device=device) # init from prior
    print("height:", input_height, "channel:", input_channels)
    x_0 = torch.randn(mc_samples, input_channels, input_height, input_height).to(gen_sde.T)
    xs, max_L, sum_beta = euler_maruyama_sampler(gen_sde, x_0, num_steps, lmbd=0, transform=transform)  # sample
    # print(x_0.shape, "x_0", np.shape(xs), "xs", np.shape(x_test), "x_test")
    x_generated_lastT = xs[-1][:mc_samples]
    x_generated_lastT = x_generated_lastT.view(mc_samples, -1).cpu().numpy()
    print(x_generated_lastT.shape)
    flag = True
    for x, _ in x_test:
        if flag:
            test_data = x
            flag = False
        else:
            test_data = torch.cat((test_data, x), 0)
    test_data = test_data.view(test_data.size(0), -1).cpu().numpy()
    kl_div = cal_kl_datasets(test_data, x_generated_lastT, 0.1)
    return kl_div, x_generated_lastT


def run_sde(args):
    with open(os.path.join('output', 'args.txt'), 'w') as out:
        out.write(json.dumps(args.__dict__, indent=4))

    prefix = "m_" + str(args.n_train) + "_rep_" + str(args.seed) + "_T_" + str(args.T0) + "_iter_" + str(args.sm_training_iter)
    #         x_train, x_test = gen_train_test_data(n_train=n_train, n_test=n_test)
    gen_sde, upper_bounds, test_kl, sm_losses, x_generated, bpds = train_sde(args, device, prefix)
    with open("./output/results/" + args.dataset + "/" + prefix + '.npy', 'wb') as f:
        #np.save(f, gen_sde)
        np.save(f, upper_bounds)
        np.save(f, test_kl)
        np.save(f, sm_losses)
        np.save(f, x_generated)
        np.save(f, bpds)


def train_sde(args, device, prefix):
    """
    X_train, X_test, T0 = 1, batch_size = 256, iterations = 10000,\
              lr = 0.001, print_every = 5000, vtype = 'rademacher', \
            #   num_steps=10, mc_samples=100, R= 0.1
    """
    ## ========== get score function model ======
    drift_q, dimx, input_channels, input_height = get_score_model(args.dataset)
    ## ========== get data loader ==========
    trainloader, testloader, transform, sub_trainset, sub_testset = gen_train_test_data(args.dataset, args.datapath, args.train_batch_size, args.test_batch_size, args.n_train,
                                                                                        args.n_test)

    ## setup diffusion model
    T = torch.nn.Parameter(torch.FloatTensor([args.T0]), requires_grad=False)
    inf_sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=T).to(device)
    gen_sde = PluginReverseSDE(inf_sde, drift_q, T, vtype=args.vtype, debias=False).to(device)

    upper_bounds = []
    test_kl = []
    sm_losses = []
    x_generated = []
    bpds = []
    # init optimizer
    optim = torch.optim.Adam(gen_sde.parameters(), lr=args.lr)
    # transform data
    logit = LogitTransform(alpha=0.05)
    if args.real:
        reverse = logit.reverse
    else:
        reverse = None

    # set output, logging
    output_path = './output/'
    model_path = output_path + 'models/' + args.dataset + "/" + prefix + '_checkpoint.pt'
    print_ = lambda s: logging(s, output_path)
    print_(f'output path: {output_path}')
    print_(str(args))
    summary_path = output_path + 'summary/' + args.dataset + '/' + prefix
    if not os.path.isdir(summary_path):
        os.makedirs(summary_path)
    writer = SummaryWriter(summary_path)
    count = 0
    load_previous = False
    step_size = args.step_size
    num_steps = int(args.T0 / step_size)
    if os.path.exists(model_path):
        # load model
        print("load previous model")
        gen_sde, optim, count, score = torch.load(model_path)
        writer.add_scalar('T', gen_sde.T.item(), count)
        writer.add_image('samples', get_grid(gen_sde, input_channels, input_height, n=4, num_steps=num_steps, transform=reverse), 0)
        load_previous = True
    else:
        writer.add_scalar('T', gen_sde.T.item(), count)
        writer.add_image('samples', get_grid(gen_sde, input_channels, input_height, n=4, num_steps=num_steps, transform=reverse), 0)
    for i in range(args.sm_training_iter):
        count += 1
        score = []
        optim.zero_grad()
        train_cnt = 0
        n = 4
        ## several batches, 16, 1-batch
        for x, _ in trainloader:
            x = x.to(device)
            if train_cnt == 0:
                train_img = x
                train_cnt += 1
            else:
                train_img = torch.cat((train_img, x), 0)
                train_cnt += 1
                if train_cnt >= n**2:
                    break
            x = x * 255 / 256 + torch.rand_like(x) / 256
            #print("before transform:", x.shape)
            if args.real:
                x, _ = logit.forward_transform(x, 0)
            #print("after transform:", x.shape)
            # train score matching loss
            loss = gen_sde.dsm(x).mean()
            loss.backward()
            optim.step()
            score.append(loss.item())
        avg_score = np.mean(score)
        if count == 1 or count % args.print_every == 0 or load_previous:
            avg_score = np.mean(score)
            y0 = train_img[:16, :].view(n, n, input_channels, input_height, input_height).permute(2, 0, 3, 1,
                                                                                                  4).contiguous().view(input_channels, n * input_height, n * input_height)
            y0 = y0.data.cpu().numpy()
            writer.add_image('train samples', y0, count)
            print("write train image successfully")
            writer.add_scalar('loss', loss.item(), count)
            writer.add_scalar('T', gen_sde.T.item(), count)

            bpd, std_err = evaluate(gen_sde, testloader, args.real, logit, dimx, args.n_test)
            bpds.append(bpd)
            writer.add_scalar('bpd', bpd, count)
            writer.add_scalar('bpd_std_err', std_err, count)
            print_(f'Iteration {count} \tBPD {bpd}')

        if count % args.checkpoint_every == 0:

            torch.save([gen_sde, optim, count, score], model_path)
            writer.add_image('samples', get_grid(gen_sde, input_channels, input_height, n=4, num_steps=num_steps, transform=reverse), count)
            print("write image successfully")

        # sample images, estimate bound at last iteration
        if (count % args.sample_every == 0) or (load_previous and count % args.sample_every == 1):
            gen_sde.eval()
            writer.add_image('samples', get_grid(gen_sde, input_channels, input_height, n=4, num_steps=num_steps, transform=reverse), count)
            print("write image successfully")
            bound = evaluate_upper_bounds(gen_sde, logit, trainloader, args.n_train, avg_score, args.R, num_steps, args.mc_samples, args.real)[0][0]
            upper_bounds.append(bound)
            sm_losses.append(avg_score)
            kl_div, x_generated_lastT = cal_generation_error(gen_sde, testloader, args.mc_samples, num_steps, input_channels, input_height, transform=reverse)

            # print(kl_div)
            test_kl.append(kl_div)
            x_generated.append(x_generated_lastT)
            print('| iter {:6d} | kl_div {} | bound {}'.format(count, kl_div, bound))
            if load_previous:
                break
            else:
                gen_sde.train()

            # generate n*n images

    return gen_sde, upper_bounds, test_kl, sm_losses, x_generated, bpds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # i/o
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--datapath', type=str, default='./datasets')
    parser.add_argument('--modelpath', type=str, default='./models')
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample_every', type=int, default=100)
    parser.add_argument('--checkpoint_every', type=int, default=10000)

    # optimization
    parser.add_argument('--T0', type=float, default=1.0, help='integration time')
    parser.add_argument('--R', type=float, default=1.0, help='sub-gaussian parameter')
    parser.add_argument('--vtype', type=str, choices=['rademacher', 'gaussian'], default='rademacher', help='random vector for the Hutchinson trace estimator')
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--sm_training_iter', type=int, default=10000, help='number of training iterations for score matching loss')
    parser.add_argument('--step_size', type=float, default=0.0001, help='generator SDE (euler maruyama sampler) sampling step size')
    parser.add_argument('--num_steps', type=int, default=10000, help='generator SDE (euler maruyama sampler) sampling steps')
    parser.add_argument('--mc_samples', type=int, default=10000, help='number of examples to estimate divergences (monte carlo sampling)')
    parser.add_argument('--n_train', type=int, default=100, help='number of train data samples')
    parser.add_argument('--n_test', type=int, default=100, help='number of test data samples')

    # model
    parser.add_argument('--real', type=eval, choices=[True, False], default=True, help='transforming the data from [0,1] to the real space using the logit function')
    parser.add_argument('--debias', type=eval, choices=[True, False], default=False, help='using non-uniform sampling to debias the denoising score matching loss')

    args = parser.parse_args()
    run_sde(args)
