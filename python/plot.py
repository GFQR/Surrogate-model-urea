import numpy as np
import matplotlib.pyplot as plt
import math

import main
import rescale

# ---------------------------------------------------------------------
def P_2D(exp_data, model, q):
    '''
    exp_data: np array with [q, theta, beta] in columns
    q is the charge value selected for the beta.vs.theta 2D plot
    '''
    mask = exp_data == q
    exp_fixq = exp_data[mask[:,0]][:,1:3] # experiment's data: beta.vs.theta

    qtheta = table_regression(model, q)

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

# ---------------------------------------------------------------------
def table_regression(model, q):
    '''
    construct, from the model, the 2-column table beta|theta for a given q (un-rescaled)
    '''
    theta_step = 0.01
    model_theta = 2*math.pi*np.arange(0, 1, theta_step)
    model_q = q*np.ones(model_theta.shape)

    if main.rescale_bool:
        qresc = rescale.std_scaling(model_q)
    else:
        qresc = model_q
    
    q2resc = np.square(qresc)
    cost = np.cos(model_theta)
    cos2t = np.cos(2*model_theta)
    sint = np.sin(model_theta)
    sin2t = np.sin(2*model_theta)
    qresc_cost = qresc*cost
    qresc_cos2t = qresc*cos2t
    qresc_sint = qresc*sint
    qresc_sin2t = qresc*sin2t

    X_col = np.column_stack((qresc, q2resc, cost, sint, cos2t, sin2t,
                             qresc_cost, qresc_sint, qresc_cos2t, qresc_sin2t))
    beta_pred = model.predict(X_col)

    qtheta = np.column_stack((model_theta, beta_pred))

    return qtheta

if __name__ == "__main__":
    None