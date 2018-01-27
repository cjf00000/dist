import numpy as np
from matplotlib import pyplot as plt

algs = ['em', 'scvb0', 'cgs', 'ds']
num_tokens = 1932365


def process(alg):
    lls = []
    with open('{}.log'.format(alg)) as f:
        lines = f.readlines()
        for line in lines:
            if line.find('Iteration') != -1:
                l = -float(line.split()[-1]) * num_tokens
                lls.append(l)

    # lls = np.array(lls)
    # lls = lls[-1] - lls
    # plt.semilogy(lls[:-1])

    plt.plot(lls)

for alg in algs:
    process(alg)

plt.legend(algs)
plt.xlim([0, 100])
plt.show()

