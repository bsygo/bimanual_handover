#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Implementation of cost function in bimanual functional regrasping paper with corresponding epsilons
scores = []
for i in np.arange(0, 1, 0.01):
    delta = [1 * i]
    epsilon = [1]
    score = 0
    for i in range(len(delta)):
        score += 1/epsilon[i]**2 * delta[i]**2 - 2/epsilon[i] * delta[i] + 1
    score = score/len(delta)
    scores.append(score)

plt.plot(scores)
plt.title("Cost Function")
plt.xlabel(r"Proportion $\delta$ to $\epsilon$ in %")
plt.ylabel("Cost Value")
plt.show()
