#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# File Name: plot_img_bound.py
# Description:
"""

from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import os
import numpy as np
from collections import OrderedDict

import pandas as pd


def load_data(dataset="cifar10", full=True):

    f_list = os.listdir("./output/results/" + dataset + "/")
    print(f_list)

    df_ub_dict = {'m': [], 't': [], 'seed': [], 'bound': []}
    df_kl_dict = {'m': [], 't': [], 'seed': [], 'kl': []}
    df_sm_dict = {'m': [], 't': [], 'seed': [], 'sm': []}
    df_x_gen_dict = {'m': [], 't': [], 'seed': [], 'x': []}
    df_bpds_dict = {'m': [], 't': [], 'seed': [], 'bpd': []}
    ub_dict = OrderedDict()
    kl_dict = OrderedDict()
    sm_dict = OrderedDict()
    x_gen_dict = OrderedDict()
    bpds_dict = OrderedDict()
    for f_name in f_list:
        str_sp = f_name.split("_")
        m = int(str_sp[1])
        t = float(str_sp[5])
        ind = int(str_sp[3])
        print(m, t, ind)
        if full:
            if (m != 60000 and dataset == "mnist") or (m != 50000 and dataset == "cifar10"):
                continue
        else:
            if m != 16:
                continue

        df_ub_dict['m'].append(m)
        df_ub_dict['seed'].append(ind)
        df_ub_dict['t'].append(t)
        df_kl_dict['m'].append(m)
        df_kl_dict['seed'].append(ind)
        df_kl_dict['t'].append(t)
        df_sm_dict['m'].append(m)
        df_sm_dict['seed'].append(ind)
        df_sm_dict['t'].append(t)
        df_x_gen_dict['m'].append(m)
        df_x_gen_dict['seed'].append(ind)
        df_x_gen_dict['t'].append(t)
        df_bpds_dict['m'].append(m)
        df_bpds_dict['seed'].append(ind)
        df_bpds_dict['t'].append(t)
        with open("./output/results/" + dataset + "/" + f_name, 'rb') as f:
            ub = np.load(f, allow_pickle=True)
            kl = np.load(f, allow_pickle=True)
            sm = np.load(f, allow_pickle=True)
            x_gen = np.load(f, allow_pickle=True)
            bpds = np.load(f, allow_pickle=True)
            df_ub_dict['bound'].append(ub[-1] / (28.0 * 28))
            df_kl_dict['kl'].append(kl[-1] / (28.0 * 28))
            df_sm_dict['sm'].append(sm[-1])
            df_x_gen_dict['x'].append(x_gen[-1])
            df_bpds_dict['bpd'].append(bpds[-1])

            if t not in ub_dict:
                ub_dict[t] = [ub[-1] / (28.0 * 28)]
                kl_dict[t] = [kl[-1] / (28.0 * 28)]
                sm_dict[t] = [sm[-1]]
                x_gen_dict[t] = [x_gen[-1]]
                bpds_dict[t] = [bpds[-1]]
            else:
                ub_dict[t].append(ub[-1] / (28.0 * 28))
                kl_dict[t].append(kl[-1] / (28.0 * 28))
                sm_dict[t].append(sm[-1])
                x_gen_dict[t].append(x_gen[-1])
                bpds_dict[t].append(bpds[-1])

    print(ub_dict.keys(), ub_dict)
    for m in sorted(ub_dict.keys()):
        m, np.mean(ub_dict[m]), np.var(ub_dict[m])

    df_ub = pd.DataFrame(df_ub_dict)
    df_kl = pd.DataFrame(df_kl_dict)
    df_bpd = pd.DataFrame(df_bpds_dict)

    return df_ub, df_kl, df_bpd


def plot_bound(df_bpd_mnist, df_bpd_cifar, df_ub_mnist, df_ub_cifar):
    # plt.figure(figsize=(8, 6))
    fig, axes = plt.subplots(2, 1, figsize=(6, 5))

    sns.color_palette("tab10")
    ax1 = sns.lineplot(x="t", y="bpd", data=df_bpd_mnist, label="bpd (mnist)", marker='o', color='green', ax=axes[0])
    ax1 = sns.lineplot(x="t", y="kl", data=df_kl_mnist, label="test data KL (mnist)", marker='o', color='orange', ax=axes[0])
    ax2 = ax1.twinx()
    ax2 = sns.lineplot(x="t", y="bound", data=df_ub_mnist, label="upper bound (mnist)", marker='o', color='tab:blue')
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax1.set(xlabel=None)
    ax2.set(xlabel=None)
    ax1.get_xaxis().set_visible(False)
    # ax1.set_visible(False)
    ax3 = sns.lineplot(x="t", y="bpd", data=df_bpd_cifar, label="bpd (cifar10) ", linestyle='--', marker='o', color='green', ax=axes[1])
    ax3 = sns.lineplot(x="t", y="kl", data=df_kl_cifar, label="test data KL (cifar10)", linestyle='--', marker='o', color='orange', ax=axes[1])
    ax4 = ax3.twinx()
    ax4 = sns.lineplot(x="t", y="bound", data=df_ub_cifar, label="upper bound (cifar10)", linestyle='--', marker='o', color='tab:blue')
    # plt.title('', fontsize=16)

    ax3.set_xlabel('Diffusion Time: T', fontsize=14)
    ax2.set_ylabel('Estimated Bound', fontsize=14)
    ax1.set_ylabel('BPD & Test Data KL', fontsize=14)
    ax4.set_ylabel('Estimated Bound', fontsize=14)
    ax3.set_ylabel('BPD & Test Data KL', fontsize=14)

    ax1.set(ylim=(5, 25))
    ax2.set(ylim=(-10, 200))
    ax4.set(ylim=(-10, 300))
    ax3.set(ylim=(5, 25))
    # ax4.set(ylim=(-25, 80))
    # ax1.set_xscale('log')
    # ax3.set_xscale('log')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc=1)
    ax2.get_legend().remove()
    h3, l3 = ax3.get_legend_handles_labels()
    h4, l4 = ax4.get_legend_handles_labels()
    ax3.legend(h3 + h4, l3 + l4, loc=4)
    ax4.get_legend().remove()
    # plt.setp(ax1.get_legend().get_texts(), fontsize='12')
    plt.tight_layout()
    plt.savefig('mnist_cifar_bound_t.pdf', format='pdf', bbox_inches='tight')


def plot_bound_full(df_bpd_mnist, df_bpd_cifar, df_ub_mnist, df_ub_cifar):
    # plt.figure(figsize=(8, 6))
    fig, axes = plt.subplots(2, 1, figsize=(6, 5))

    sns.color_palette("tab10")
    ax1 = sns.lineplot(x="t", y="bpd", data=df_bpd_mnist, label="bpd (mnist)", marker='o', color='green', ax=axes[0])
    ax2 = ax1.twinx()
    ax2 = sns.lineplot(x="t", y="bound", data=df_ub_mnist, label="upper bound (mnist)", marker='o', color='tab:blue')
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax1.set(xlabel=None)
    ax2.set(xlabel=None)
    ax1.get_xaxis().set_visible(False)
    # ax1.set_visible(False)
    ax3 = sns.lineplot(x="t", y="bpd", data=df_bpd_cifar, label="bpd (cifar10)", linestyle='--', marker='o', color='green', ax=axes[1])
    ax4 = ax3.twinx()
    ax4 = sns.lineplot(x="t", y="bound", data=df_ub_cifar, label="upper bound (cifar10)", linestyle='--', marker='o', color='tab:blue')
    # plt.title('', fontsize=16)

    ax3.set_xlabel('Diffusion Time: T (log scale)', fontsize=14)
    ax2.set_ylabel('Estimated Bound', fontsize=14)
    ax1.set_ylabel('BPD', fontsize=14)
    ax4.set_ylabel('Estimated Bound', fontsize=14)
    ax3.set_ylabel('BPD', fontsize=14)

    ax1.set(ylim=(1, 10))
    ax2.set(ylim=(-80, 200))
    ax3.set(ylim=(3, 20))
    ax4.set(ylim=(-25, 80))
    ax1.set_xscale('log')
    ax3.set_xscale('log')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc=1)
    ax2.get_legend().remove()
    h3, l3 = ax3.get_legend_handles_labels()
    h4, l4 = ax4.get_legend_handles_labels()
    ax3.legend(h3 + h4, l3 + l4, loc=1)
    ax4.get_legend().remove()
    # plt.setp(ax1.get_legend().get_texts(), fontsize='12')
    plt.tight_layout()
    plt.savefig('mnist_cifar_bound_t_f.pdf', format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    df_ub_mnist, df_kl_mnist, df_bpd_mnist = load_data("mnist", full=False)
    df_ub_cifar, df_kl_cifar, df_bpd_cifar = load_data("cifar10", full=False)
    plot_bound(df_bpd_mnist, df_bpd_cifar, df_ub_mnist, df_ub_cifar)
    df_ub_mnist, df_kl_mnist, df_bpd_mnist = load_data("mnist", full=True)
    df_ub_cifar, df_kl_cifar, df_bpd_cifar = load_data("cifar10", full=True)
    plot_bound_full(df_bpd_mnist, df_bpd_cifar, df_ub_mnist, df_ub_cifar)
