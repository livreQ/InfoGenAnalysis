#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Sun Mar 16 16:43:50 2025
# File Name: plot_toy_bound.py
# Description:
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from collections import OrderedDict


def load_data(dataset="toy", variant="t"):

    f_list = os.listdir("./output/results/" + dataset + "/")
    print(f_list)

    df_ub_dict = {'m': [], 't': [], 'seed': [], 'bound': []}
    df_kl_dict = {'m': [], 't': [], 'seed': [], 'kl': []}
    df_sm_dict = {'m': [], 't': [], 'seed': [], 'sm': []}
    df_x_gen_dict = {'m': [], 't': [], 'seed': [], 'x': []}

    ub_dict = OrderedDict()
    kl_dict = OrderedDict()
    sm_dict = OrderedDict()
    x_gen_dict = OrderedDict()
    for f_name in f_list:
        str_sp = f_name.split("_")
        m = int(str_sp[1])
        t = float(str_sp[5])
        ind = int(str_sp[3])
        print(m, t, ind)
        if variant == "t":
            if m != 200:  # fix m to 200
                continue
        else:
            if t != 1.0:  # fix t to 1.0
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
        with open("./output/results/" + dataset + "/" + f_name, 'rb') as f:
            ub = np.load(f, allow_pickle=True)
            kl = np.load(f, allow_pickle=True)
            sm = np.load(f, allow_pickle=True)
            x_gen = np.load(f, allow_pickle=True)
            df_ub_dict['bound'].append(ub[-1][0])
            df_kl_dict['kl'].append(kl[-1])
            df_sm_dict['sm'].append(sm[-1])
            df_x_gen_dict['x'].append(x_gen[-1])
            if t not in ub_dict:
                ub_dict[t] = [ub[-1]]
                kl_dict[t] = [kl[-1]]
                sm_dict[t] = [sm[-1]]
                x_gen_dict[t] = [x_gen[-1]]
            else:
                ub_dict[t].append(ub[-1])
                kl_dict[t].append(kl[-1])
                sm_dict[t].append(sm[-1])
                x_gen_dict[t].append(x_gen[-1])

    for m in sorted(ub_dict.keys()):
        m, np.mean(ub_dict[m]), np.var(ub_dict[m])
    df_ub = pd.DataFrame(df_ub_dict)
    df_kl = pd.DataFrame(df_kl_dict)
    return df_kl, df_ub


def plot_bound_t(df_kl, df_ub):
    plt.figure(figsize=(8, 6))
    ax1 = sns.lineplot(x="t", y="kl", label="test data KL", marker='*', data=df_kl, color='orange')
    ax2 = plt.twinx()
    ax2 = sns.lineplot(x="t", y="bound", data=df_ub, label="upper bound", marker='o')

    ax1.set_xlabel('Diffusion Time: T', fontsize=16)
    ax2.set_ylabel('Estimated Bound', fontsize=16)
    ax1.set_ylabel('Test Data KL', fontsize=16)

    ax2.set(ylim=(6, 20))
    ax1.set(ylim=(0.2, 2))
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc=2)
    ax2.get_legend().remove()
    plt.setp(ax1.get_legend().get_texts(), fontsize='16')
    plt.tight_layout()
    plt.savefig('bound_t.pdf', format='pdf', bbox_inches='tight')


def plot_bound_m(df_kl_m, df_ub_m):

    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(x="m", y="kl", label="test data KL", marker='*', data=df_kl_m, color='orange')
    ax = sns.lineplot(x="m", y="bound", label="upper bound", marker='o', data=df_ub_m)
    plt.setp(ax.get_legend().get_texts(), fontsize='16')
    plt.xlabel('Number of Examples: m', fontsize=16)
    plt.ylabel('Estimated Value', fontsize=16)
    plt.tight_layout()
    plt.savefig('bound_m.pdf', format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    df_kl, df_ub = load_data(variant="t")
    plot_bound_t(df_kl, df_ub)
    df_kl_m, df_ub_m = load_data(variant="m")
    plot_bound_m(df_kl_m, df_ub_m)
