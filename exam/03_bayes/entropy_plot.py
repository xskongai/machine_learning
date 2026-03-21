#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: entropy_plot
"""
import numpy as np
import matplotlib.pyplot as plt


def shannon_entropy():
    # Compute Bernoulli entropy H(p) in nats
    p = np.linspace(0.001, 0.999, 1000)
    H_nats = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    # Value at p = 0.25
    p0 = 0.25
    H_p0 = -(p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))

    plt.figure(figsize=(8, 4.5))
    plt.plot(p, H_nats, linewidth=2)
    plt.scatter([p0], [H_p0])
    plt.text(p0 + 0.02, H_p0, f"p={p0}\nH≈{H_p0:.3f} nats", va='center')
    plt.xlabel('p')
    plt.ylabel('H(p) (nats)')
    plt.title('Bernoulli entropy H(p) in nats')
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, np.log(2) + 0.05)  # max at ln 2
    plt.show()

def  bits_nats():
    p = np.linspace(0.001, 0.999, 1000)

    # Entropy in bits
    H_bits = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    # Entropy in nats
    H_nats = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    plt.figure(figsize=(8, 4.5))
    plt.plot(p, H_bits, label='H(p) in bits', linewidth=2)
    plt.plot(p, H_nats, label='H(p) in nats', linewidth=2)

    plt.xlabel('p')
    plt.ylabel('Entropy')
    plt.title('Bernoulli entropy in bits vs nats')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)
    plt.show()


def main():
    shannon_entropy()
    bits_nats()


if __name__ == "__main__":
    main()
