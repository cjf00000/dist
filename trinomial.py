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


num_particles = 1000

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
    z2 = np.expand_dims(et / (2 - et) * (N - y), 1)
    z3 = np.expand_dims((2 - 2 * et) / (2 - et) * (N - y), 1)
    z = np.concatenate([z1, z2, z3], 2)
    print('Z shape = {}'.format(z.shape))
    return z

def get_zs(theta):
    z1 = np.expand_dims(np.array([y] * theta.shape[0]), 1)
    z2 = np.expand_dims(theta / (2 - theta) * (N - y), 1)
    z3 = np.expand_dims((2 - 2 * theta) / (2 - theta) * (N - y), 1)
    z = np.hstack([z1, z2, z3])
    return z

learning_rate = 0.01
z     = np.tile(np.expand_dims(np.array([0, 0, N]), 0), [num_particles, 1]).astype(np.float32)
sqr_errors = []
# for i in range(3000):
for i in range(400):
    theta = get_thetas(z)
    zhat  = get_zhat(theta)
    z     = (1-learning_rate)*z + learning_rate*zhat
    error = np.square(z-z0).sum(1).mean()
    sqr_errors.append(error)

# E(z-z0)^2
error = np.square(z-z0).sum(1).mean()
print('Error = {}'.format(error))

zhat = get_zhat(theta)
error_zhat_z0 = np.square(zhat-z0).sum(1).mean()
print('Error (zhat - z0) = {}'.format(error_zhat_z0))

zbar = get_zs(theta)
var_z = np.square(zhat-zbar).sum(1).mean()
error_zbar_z0 = np.square(zbar-z0).sum(1).mean()
print('{} {} {} {}'.format(error_zhat_z0, var_z, error_zbar_z0, var_z+error_zbar_z0))

next_z = (1-learning_rate)*z + learning_rate*zhat
error_nextz_z0 = np.square(next_z-z0).sum(1).mean()
print( (1-learning_rate)**2*error + learning_rate**2*error_zhat_z0 )
print('Error (fz-z0) = {}'.format(error_nextz_z0))

# estimated_sqr_errors = sqr_errors[:10]
# for i in range(2990):
#     err     = estimated_sqr_errors[-1]
#     new_err = ((1-learning_rate)**2+learning_rate**2*r)*err + learning_rate**2*v
#     estimated_sqr_errors.append(new_err)
#
# print( (1-learning_rate)**2 + learning_rate**2*r )
# print( sqr_errors[11] / sqr_errors[10] )
#
# # print('Real contraction ratio = {}'.format( (T-(1-learning_rate)**2)/learning_rate**2 ))
#
# plt.semilogy(sqr_errors)
# plt.semilogy(estimated_sqr_errors)
# # plt.plot(thetas)
# plt.show()