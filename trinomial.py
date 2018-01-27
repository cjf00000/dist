import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(1)

theta = 0.3
N     = 1000000
z1    = N * theta // 2
z2    = N * theta // 2
z3    = N - z1 - z2
z0    = np.array([z1, z2, z3], dtype=np.float32)
y     = z1
y0    = y
theta0 = theta

# theta = 0
# thetas = []
# for i in range(100):
#     est_z1 = y
#     est_z2 = theta / (2 - theta) * (N - y)
#     est_z3 = (2 - 2*theta) / (2 - theta) * (N - y)
#     theta = (est_z1 + est_z2) / (est_z1 + est_z2 + est_z3)
#     print(est_z1, est_z2, est_z3, theta)
#     thetas.append(theta)
#
# thetas = np.array(thetas)
# error  = np.abs(thetas - theta0)
#
# rate = 2 * (N - y) / (2 - theta0)**2 / N
# print(rate)
# print(error[11] / error[10])
#
# plt.semilogy(error)
# plt.show()

def get_theta(z):
    return (z[0]+z[1])/(z[0]+z[1]+z[2])

def get_z(theta):
    est_z1 = y
    est_z2 = theta / (2 - theta) * (N - y)
    est_z3 = (2 - 2*theta) / (2 - theta) * (N - y)
    return np.array([est_z1, est_z2, est_z3], np.float32)

def sqrnorm(x):
    return np.square(x).sum()

# Estimate r = E (bar z-z0)^2 / E (z - z0)^2
z     = np.array([z1, z2+200, z3-200], np.float32)
theta = get_theta(z)
bar_z = get_z(theta)
print(z)
print(bar_z)
r     = sqrnorm(bar_z-z0) / sqrnorm(z-z0)
print('Contraction ratio = {}'.format(r))

# Estimate v = E(hat z - bar z)^2
errors = []
for iter in range(1000):
    if np.random.random() < theta0 / 2:
        y = N
    else:
        y = 0

    hat_z = np.array([y, theta/(2-theta)*(N-y), (2-2*theta)/(2-theta)*(N-y)])
    errors.append(sqrnorm(hat_z - bar_z))
v = np.mean(errors)
print('Variance = {}'.format(v))


num_particles = 100

def get_thetas(z):
    return (z[:,0]+z[:,1]) / (z[:,0]+z[:,1]+z[:,2])

def get_zhat(theta):
    y = N * (np.random.rand(num_particles) < theta0/2)
    z1 = np.expand_dims(y, 1)
    z2 = np.expand_dims(theta / (2 - theta) * (N - y), 1)
    z3 = np.expand_dims((2 - 2 * theta) / (2 - theta) * (N - y), 1)
    z = np.hstack([z1, z2, z3])
    return z

def get_zhats(theta, num_samples=100):
    # Shape: [n_p n_s dim]

    # For each theta, draw num_samples ys
    y  = N * (np.random.rand(num_particles, num_samples) < theta0/2)
    et = np.expand_dims(theta, 1)

    z1 = np.expand_dims(y, 2)
    z2 = np.expand_dims(et / (2 - et) * (N - y), 2)
    z3 = np.expand_dims((2 - 2 * et) / (2 - et) * (N - y), 2)

    z = np.concatenate([z1, z2, z3], 2)
    return z

def get_zs(theta):
    y = y0
    z1 = np.expand_dims(np.array([y] * theta.shape[0]), 1)
    z2 = np.expand_dims(theta / (2 - theta) * (N - y), 1)
    z3 = np.expand_dims((2 - 2 * theta) / (2 - theta) * (N - y), 1)
    z = np.hstack([z1, z2, z3])
    return z

def get_dist(z1, z2):
    if len(z1.shape)==2:
        z1 = np.expand_dims(z1, 1)
    if len(z2.shape)==2:
        z2 = np.expand_dims(z2, 1)
    return np.square(z1-z2).sum(-1).mean()

learning_rate = 0.01
z     = np.tile(np.expand_dims(np.array([0, 0, N]), 0), [num_particles, 1]).astype(np.float32)
sqr_errors = []
for i in range(3000):
    if i % 100 == 0:
        print(i)
    theta = get_thetas(z)
    zhat  = get_zhat(theta)
    zhats = get_zhats(theta)
    z     = (1-learning_rate)*z + learning_rate*zhat
    zs    = (1-learning_rate)*np.expand_dims(z, 1) + learning_rate*zhats
    error = get_dist(zs, z0)
    sqr_errors.append(error)


# # E(z-z0)^2
# error = np.square(z-z0).sum(1).mean()
# print('Error = {}'.format(error))
#
# zhat = get_zhats(theta)
# error_zhat_z0 = get_dist(zhat, z0)
# print('Error (zhat - z0) = {}'.format(error_zhat_z0))
#
# zbar = get_zs(theta)
# var_z = get_dist(zhat, zbar)
# error_zbar_z0 = get_dist(zbar, z0)
# ediff = np.mean(zhat - np.expand_dims(zbar, 1), 1)
# print('{} = {} + {} = {}'.format(error_zhat_z0, var_z, error_zbar_z0, var_z+error_zbar_z0))
# print('E(zbar - z0)^2 ({}) = r E(z-z0)^2 = ({})'.format(error_zbar_z0,
#                                                         r*error))
#
# next_z = (1-learning_rate)*np.expand_dims(z, 1) + learning_rate*zhat
# error_nextz_z0 = get_dist(next_z, z0)
# pred_error = ((1-learning_rate)**2+learning_rate**2*r)*error + learning_rate**2*var_z + \
#              2*(1-learning_rate)*learning_rate*np.sqrt(r)*error
# print('E(nextz - z0)^2 ({}) = {}'.format(error_nextz_z0,
#                                          pred_error))
#
# print('E(nextz - z0)^2 ({}) = {}'.format(error_nextz_z0,
#                                          (1-learning_rate)**2*error + learning_rate**2*error_zhat_z0))
#
# term1 = (1-learning_rate)**2*error
# term2 = learning_rate**2*error_zhat_z0
# E_z_z0_zhat_z0 = 2*((np.expand_dims(z, 1)-z0) * (zhat - z0)).sum(1).mean()
# term3 = E_z_z0_zhat_z0 * (1-learning_rate) * learning_rate
# print(term1, term2, term3, term1+term2+term3, error_nextz_z0)

estimated_sqr_errors = sqr_errors[:100]
for i in range(2900):
    err     = estimated_sqr_errors[-1]
    new_err = ((1-learning_rate)**2 +
               learning_rate**2*r +
               2*learning_rate*(1-learning_rate)*np.sqrt(r))*err + learning_rate**2*v
    estimated_sqr_errors.append(new_err)

# print(estimated_sqr_errors[500], sqr_errors[500])

# print('Real contraction ratio = {}'.format( (T-(1-learning_rate)**2)/learning_rate**2 ))

plt.semilogy(sqr_errors)
plt.semilogy(estimated_sqr_errors)
# plt.plot(thetas)
# plt.show()

# 1/T learning rate
learning_rate = 0.01
z     = np.tile(np.expand_dims(np.array([0, 0, N]), 0), [num_particles, 1]).astype(np.float32)
sqr_errors = []
lrs = []
for i in range(3000):
    learning_rate = 1.0 / (i + 2)
    if i % 100 == 0:
        print(i)
    theta = get_thetas(z)
    zhat  = get_zhat(theta)
    zhats = get_zhats(theta)
    z     = (1-learning_rate)*z + learning_rate*zhat
    zs    = (1-learning_rate)*np.expand_dims(z, 1) + learning_rate*zhats
    error = get_dist(zs, z0)
    sqr_errors.append(error)

plt.semilogy(sqr_errors)

# 1/sqrt(T) learning rate
learning_rate = 0.01
z     = np.tile(np.expand_dims(np.array([0, 0, N]), 0), [num_particles, 1]).astype(np.float32)
sqr_errors = []
lrs = []
for i in range(3000):
    learning_rate = 1.0 / np.sqrt(i + 2)
    if i % 100 == 0:
        print(i)
    theta = get_thetas(z)
    zhat  = get_zhat(theta)
    zhats = get_zhats(theta)
    z     = (1-learning_rate)*z + learning_rate*zhat
    zs    = (1-learning_rate)*np.expand_dims(z, 1) + learning_rate*zhats
    error = get_dist(zs, z0)
    sqr_errors.append(error)

plt.semilogy(sqr_errors)


# Best learning rate
learning_rate = 0.01
z     = np.tile(np.expand_dims(np.array([0, 0, N]), 0), [num_particles, 1]).astype(np.float32)
sqr_errors = []
lrs = []
for i in range(3000):
    if i % 100 == 0:
        print(i)
    theta = get_thetas(z)
    zhat  = get_zhat(theta)
    zhats = get_zhats(theta)
    z     = (1-learning_rate)*z + learning_rate*zhat
    zs    = (1-learning_rate)*np.expand_dims(z, 1) + learning_rate*zhats
    error = get_dist(zs, z0)
    sqr_errors.append(error)
    learning_rate = (1 - np.sqrt(r)) * error / ((1-np.sqrt(r))**2*error + v)

    if i % 100 == 0:
        p = learning_rate
        term1 = ((1-p)**2 + p**2*r + 2*p*(1-p)*np.sqrt(r)) * error
        term2 = p**2 * v
        print(term1, term2)

    lrs.append(learning_rate)

plt.semilogy(sqr_errors)
plt.legend(('constant', 'constant (predicted)', '1/T', '1/sqrt(T)', 'best'))
plt.savefig('plt.pdf')


fig, ax = plt.subplots()
ax.plot(1.0 / np.array(lrs))
fig.savefig('lr.pdf')