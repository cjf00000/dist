import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

n_dims = 100000
f_s    = 10000
s_s    = np.arange(100, 10100, 100)
dist   = np.random.random(n_dims)
dist   = dist / dist.sum()
samples = np.random.multinomial(f_s, dist)

print('Entropy = {}'.format(-(dist*np.log(dist)).sum()))

ents = []
for i in s_s:
    s = (np.random.multinomial(i, dist) + 1).astype(np.float32)
    pp = s / s.sum()

    ents.append( (samples*np.log(pp)).sum() )

plt.plot(s_s, ents)
plt.show()