import numpy as np
import matplotlib.pyplot as plt

import main
import utils

# ---------------------------------------------------------------------
def P_2D(exp_data, model, q_scalar, qresc_scalar):
    '''
    exp_data: np array with [q, theta, beta] in columns
    q is the charge value selected for the beta.vs.theta 2D plot
    '''
    mask = exp_data == q_scalar
    exp_fixq = exp_data[mask[:,0]][:,1:3] # experiment's data: beta.vs.theta

    qtheta = utils.table_regression(model, qresc_scalar) # determines model dense table theta|beta

    plt.figure(figsize=(5,3))
    plt.scatter(exp_fixq[:,0], exp_fixq[:,1], color='red', marker='o', label='experiment')
    plt.plot(qtheta[:,0], qtheta[:,1], color='blue', label = 'regression')
    plt.xlabel('angle [rad]')
    plt.ylabel('beta_YYY [arb]')
    plt.title('Hyperpolarizability')
    plt.grid(True)
    plt.legend()
    plt.show()

    return 

if __name__ == "__main__":
    None