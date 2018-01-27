import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

a = 0.1
r = 1

rs = []

for i in range(1000):
    r = (1-a*r) * r
    rs.append(1.0 / r)
    print(r)


plt.plot(rs)
plt.savefig('array.pdf')