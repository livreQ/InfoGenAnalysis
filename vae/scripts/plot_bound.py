#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Tue Nov 19 03:02:33 2024
# File Name: plot.py
# Description:
"""

import argparse
import gzip
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import pandas as pd
from typing import Dict

from analysis_utils import Line
from analysis_utils import dict2tex


def parse_full(filename: str) -> Dict[int, Dict[int, float]]:

    params = defaultdict(set)
    meta_keys_max = {"dataset": 1, "model": 1, "seed": 1}

    with gzip.open(filename, "rb") as fp:
        contents = fp.read()
        data = json.loads(contents.decode("utf-8"))

    metadata = data["meta"]
    results = data["results"]

    print(results.keys())
    print(results["losses"].keys())
    # Ensure we're not mixing datasets/models/seeds
    for key in meta_keys_max:
        params[key].add(metadata[key])
        assert len(params[key]) <= meta_keys_max[key]

    train_loss = results["losses"]["train"]
    test_loss = results["losses"]["test"]
    logpx = results["logpxs"]

    return train_loss, test_loss, logpx


def show_plot(epochs, lr3_train, lr3_test, logpx, M3):

    m = 60000
    ratio = 1
    loss_train = []
    KLD_train = []
    REC_train = []
    MI_train = []
    loss_test = []
    REC_test = []
    logpx_test = []
    our_bound = []
    previous_bound = []
    log_idx = []
    burn = 5
    for e in epochs:
        l_train = lr3_train[e - burn - 1][0]
        l_test = lr3_test[e - burn - 1][0]
        kld_train = lr3_train[e - burn - 1][1]
        rec_train = lr3_train[e - burn - 1][2]
        rec_test = lr3_test[e - burn - 1][2]
        mi_train = lr3_train[e - burn - 1][3]
        loss_train.append(l_train)
        KLD_train.append(kld_train)
        REC_train.append(rec_train)
        REC_test.append(rec_test)
        MI_train.append(np.sqrt(mi_train))
        loss_test.append(rec_test + np.sqrt(l_test - rec_test))
        our_bound.append(rec_train + (1 + ratio) * np.sqrt(kld_train) + (1 + ratio) * np.sqrt(1 / m) * np.sqrt(mi_train))
        previous_bound.append(rec_train + np.sqrt(kld_train))
        idx = e - 1
        # print(idx)
        if idx % 5 == 0:
            log_idx.append(e - burn - 1)
            logpx_test.append(-logpx[str(idx)]["test"][str(idx)])

    # f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    # ax1.plot(x, y)
    palette = sns.color_palette()
    df_test_loss = pd.DataFrame({"Epochs": epochs, "Loss": loss_test})
    df_o_bound = pd.DataFrame({"Epochs": epochs, "Bound Value": our_bound})
    df_p_bound = pd.DataFrame({"Epochs": epochs, "Bound Value": previous_bound})
    plt.figure(figsize=(8, 6))
    ax1 = sns.lineplot(x="Epochs", y="Loss", data=df_test_loss, label="test loss", color='green')
    ax1 = sns.lineplot(x="Epochs", y="Bound Value", label="our bound", data=df_o_bound, color='orange')
    ax1 = sns.lineplot(x="Epochs", y="Bound Value", label="previous bound", data=df_p_bound)
    # plt.show()
    plt.setp(ax1.get_legend().get_texts(), fontsize='16')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Estimated Bound', fontsize=16)
    # plt.tight_layout()
    # plt.plot(epochs, our_bound, c="tab:blue", label="$\eta = 10^{-3}$, Our bound")
    # plt.plot(epochs, previous_bound, c="tab:blue", ls="--", label="$\eta = 10^{-3}$, previous bound",)
    # plt.plot(epochs, np.asarray(loss_test), c="tab:orange", label="$\eta = 10^{-3}$, test loss",)
    # print(logpx_test)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()

    plt.tight_layout()
    plt.savefig('bound_vae.pdf', format='pdf', bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    print("length:", len(np.quantile(M3, 0.999, axis=0)), len(epochs))
    epochs1 = np.array(list(range(5, 105, 5)))
    df_mem = pd.DataFrame({"Epochs": epochs1, "Score": np.quantile(M3, 0.95, axis=0)})
    df_mem2 = pd.DataFrame({"Epochs": epochs1, "Score": np.quantile(M3, 0.99, axis=0)})
    df_mi = pd.DataFrame({"Epochs": epochs, "Score": MI_train})

    # ax1 = sns.lineplot(x="Epochs", y="Score", data=df_mem2, label="memorization score (q = 0.99)", color='orange')
    ax1 = sns.lineplot(x="Epochs", y="Score", data=df_mem, label="memorization score (q = 0.95)", color='green')
    ax2 = plt.twinx()
    ax2 = sns.lineplot(x="Epochs", y="Score", label="mutual information", ls="--", data=df_mi)
    # ax2 = sns.lineplot(x="t", y="bound", data=df_ub, label="upper bound", marker='o')
    #     plt.plot(
    #     epochs,
    #     np.quantile(M3, 0.999, axis=0),
    #     ls="--",
    #     color="tab:blue",
    #     label="$\\eta = 10^{-3}$, q = 0.999",
    # )
    ax1.set_xlabel('Epochs', fontsize=16)
    ax2.set_ylabel('Estimated Mutual Information', fontsize=16)
    ax1.set_ylabel('Memorization Score', fontsize=16)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc=0)
    ax2.get_legend().remove()
    plt.setp(ax1.get_legend().get_texts(), fontsize='16')
    plt.tight_layout()
    # plt.show()
    plt.savefig('mem_mi.pdf', format='pdf', bbox_inches='tight')


def main():
    lr3_file = "./results/binarized_mnist_extra_lr3/results/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed42_full_repeat0.json.gz"
    lr3_train, lr3_test, logpx = parse_full(lr3_file)
    # print(logpx.keys())
    burn = 5
    L = len(lr3_train)
    epochs = list(range(burn + 1, L + 1))
    print(epochs)
    lr3_train = lr3_train[burn:]
    # KLD_train, REC_train, MI_train
    # KLD_test, REC_test, MI_test
    lr3_test = lr3_test[burn:]
    lr3_sum = "./summaries/mem_bmnist_lr3.npz"
    bmnist3 = np.load(lr3_sum)
    M3 = bmnist3["M"]

    show_plot(epochs, lr3_train, lr3_test, logpx, M3)


if __name__ == "__main__":
    main()
