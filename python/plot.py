import numpy as np
import matplotlib.pyplot as plt

import main
import utils

# ---------------------------------------------------------------------
def P_2D(dalton_data, model, q_scalar, qresc_scalar):
    '''
    dalton_data: np array with [q, theta, beta] in columns
    q is the charge value selected for the beta.vs.theta 2D plot
    '''
    mask = dalton_data == qresc_scalar
    dalton_data_fixq = dalton_data[mask[:,0]][:,1:3] # Dalton's data: beta.vs.theta

    qtheta = utils.table_regression_1D(model, qresc_scalar) # determines model dense table theta|beta

    plt.figure(figsize=(5,3))
    plt.scatter(dalton_data_fixq[:,0], dalton_data_fixq[:,1], color='red', marker='o', label='experiment')
    plt.plot(qtheta[:,0], qtheta[:,1], color='blue', label = 'regression')
    plt.xlabel('angle [rad]')
    plt.ylabel('beta_YYY [arb]')
    plt.title('Hyperpolarizability')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    return 

# ---------------------------------------------------------------------
def P_3D(exp_data, surface):
    '''
    exp_data: np array with [q, theta, beta] from Dalton
    temp_qresc_m: np array with q values
    temp_theta_m: np array with theta values
    beta_pred: np array with predicted beta values
    '''
    temp_qresc_m, temp_theta_m, beta_pred = surface

    q = temp_qresc_m
    theta = temp_theta_m
    beta = beta_pred
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(q, theta, beta)
    ax.set_xlabel('charge q [e]')
    ax.set_ylabel('angle [rad]')
    ax.set_zlabel('beta')
    plt.show(block=False)

    return


if __name__ == "__main__":
    None