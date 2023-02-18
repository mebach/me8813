
#####################################################
##  ME8813ML Homework 1:
##  Implement a quasi-Newton optimization method for data fitting
#####################################################
import numpy as np
import matplotlib.pyplot as plt
########################################################


def DFP_fit(x, y, epsilon=0.0001):
    # Initialize quantities
    p_prev = np.array([[0],
                       [0],
                       [0],
                       [0]])
    B = np.eye(4)
    L_prev = 999
    L_cur= L(x, y, p_prev)

    # Loop
    while np.abs(L_cur - L_prev) > epsilon:

        di = -B @ gradL(x, y, p_prev)
        aa = min_linesearch(L, x, y, p_prev, di)
        p = p_prev + aa*di
        delta_p = p - p_prev
        delta_g = gradL(x, y, p) - gradL(x, y, p_prev)
        B = B + ((delta_p @ delta_p.T))/(delta_p.T @ delta_g) - ((B @ delta_g)@np.transpose(B @ delta_g))/((np.transpose(delta_g)@B@delta_g))
        p_prev = p
        L_prev = L_cur
        L_cur = L(x, y, p)

    return p


def min_linesearch(func, x, y, p_prev, di):
    step = 0.0001
    while func(x, y, p_prev) > func(x, y, p_prev + step*di):
        step += 0.0001

    return step

def L(x, y, p):
    return np.sum(np.square(
        p[0] + p[1] * np.cos(2 * np.pi * x) + p[2] * np.cos(4 * np.pi * x) + p[3] * np.cos(6 * np.pi * x) - y))


def gradL(x, y, p):
    return np.array([[np.sum(2 * (p[0] + p[1] * np.cos(2 * np.pi * x) + p[2] * np.cos(4 * np.pi * x) + p[3] * np.cos(6 * np.pi * x) - y))],
                      [np.sum(2 * (p[0] + p[1] * np.cos(2 * np.pi * x) + p[2] * np.cos(4 * np.pi * x) + p[3] * np.cos(
                          6 * np.pi * x) - y) * (np.cos(2 * np.pi * x)))],
                      [np.sum(2 * (p[0] + p[1] * np.cos(2 * np.pi * x) + p[2] * np.cos(4 * np.pi * x) + p[3] * np.cos(
                          6 * np.pi * x) - y) * (np.cos(4 * np.pi * x)))],
                      [np.sum(2 * (p[0] + p[1] * np.cos(2 * np.pi * x) + p[2] * np.cos(4 * np.pi * x) + p[3] * np.cos(
                          6 * np.pi * x) - y) * (np.cos(6 * np.pi * x)))]])


########################################################
# Fixing random state for reproducibility
np.random.seed(19680801)
dx = 0.1
x_lower_limit = 0
x_upper_limit = 40
x = np.arange(x_lower_limit, x_upper_limit, dx)
data_size = len(x)                                 # data size
noise = np.random.randn(data_size)                 # white noise
# Original dataset
y = 2.0 + 3.0*np.cos(2*np.pi*x) + 1.0*np.cos(6*np.pi*x) + noise
###########################################
p = DFP_fit(x, y)
print(p)
y_pred = p[0] + p[1] * np.cos(2 * np.pi * x) + p[2] * np.cos(4 * np.pi * x) + p[3] * np.cos(6 * np.pi * x)
###########################################
fig, axs = plt.subplots(2, 1)
axs[0].plot(x, y)
axs[0].set_xlim(x_lower_limit, x_upper_limit)
axs[0].set_xlabel('x')
axs[0].set_ylabel('observation')
axs[0].grid(True)
#########################################
## Plot the predictions from your fitted model here
axs[1].plot(x, y_pred)
axs[1].set_xlim(x_lower_limit, x_upper_limit)
axs[1].set_xlabel('x')
axs[1].set_ylabel('model prediction')
fig.tight_layout()
plt.show()




