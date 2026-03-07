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

    '''
    if main.rescale_bool:
        qresc = rescale.std_scaling(model_q)
    else:
        qresc = model_q
    '''
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

    # determine the new X to feed the Ridge model and get beta_pred for 
    # all those points
    X_col = np.column_stack((qresc, q2resc, cost, sint, cos2t, sin2t,
                             qresc_cost, qresc_sint, qresc_cos2t, qresc_sin2t))

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