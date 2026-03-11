import numpy as np
import math

import main
import rescale

# ---------------------------------------------------------------------
def table_regression(model, qresc_scalar):
    '''
    construct, from the model, the 2-column table beta|theta for a given q (rescaled)
    '''
    theta_step = 0.01
    model_theta = 2*math.pi*np.arange(0, 1, theta_step)
    model_q = qresc_scalar*np.ones(model_theta.shape)

    q2 = np.square(model_q)
    cost = np.cos(model_theta)
    cos2t = np.cos(2*model_theta)
    sint = np.sin(model_theta)
    sin2t = np.sin(2*model_theta)
    q_cost = model_q*cost
    q_cos2t = model_q*cos2t
    q_sint = model_q*sint
    q_sin2t = model_q*sin2t
    # new terms (March 8, 2026)
    q2_cost = model_q*model_q*cost
    q2_sint = model_q*model_q*sint
    cos3t = np.cos(3*model_theta)
    sin3t = np.sin(3*model_theta)
    q_cos3t = model_q*cos3t
    q_sin3t = model_q*sin3t

    # determine the new X to feed the Ridge model and get beta_pred for 
    # all those points
    X_col = np.column_stack((model_q, q2, cost, sint, cos2t, sin2t,
                             q_cost, q_sint, q_cos2t, q_sin2t,
                             q2_cost, q2_sint, cos3t, sin3t, q_cos3t,
                             q_sin3t))

    beta_pred = model.predict(X_col)

    qtheta = np.column_stack((model_theta, beta_pred))

    return qtheta

# ---------------------------------------------------------------------
def max_find(qtheta):
    '''
    find the maximum of beta for a given q
    '''
    index_max = np.argmax(qtheta[:,1])
    theta_max = qtheta[index_max,0]
    beta_max = qtheta[index_max,1]

    return theta_max, beta_max


if __name__ == "__main__":
    None