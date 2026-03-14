import numpy as np
import math

import main
import rescale

# ---------------------------------------------------------------------
def table_regression_1D(model, qresc_scalar):
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
def table_regression_2D(model, qresc):
    '''
    construct, from the model, the 3-column table beta|q|theta
    '''
    theta_step = 0.01
    model_theta = 2*math.pi*np.arange(0, 1, theta_step)
    qresc_max = qresc.max()
    qresc_min = qresc.min()
    qresc_step = (qresc_max - qresc_min) / (model_theta.size - 1) # same number of q and theta points
    model_qresc = np.arange(qresc_min, qresc_max, qresc_step)

    temp_qresc_m, temp_theta_m = np.meshgrid(model_qresc, model_theta)
    model_qresc_f = temp_qresc_m.flatten()
    model_theta_f = temp_theta_m.flatten()

    q2 = np.square(model_qresc_f)
    cost = np.cos(model_theta_f)
    cos2t = np.cos(2*model_theta_f)
    sint = np.sin(model_theta_f)
    sin2t = np.sin(2*model_theta_f)
    q_cost = model_qresc_f*cost
    q_cos2t = model_qresc_f*cos2t
    q_sint = model_qresc_f*sint
    q_sin2t = model_qresc_f*sin2t
    # new terms (March 8, 2026)
    q2_cost = model_qresc_f*model_qresc_f*cost
    q2_sint = model_qresc_f*model_qresc_f*sint
    cos3t = np.cos(3*model_theta_f)
    sin3t = np.sin(3*model_theta_f)
    q_cos3t = model_qresc_f*cos3t
    q_sin3t = model_qresc_f*sin3t

    # determine the new X to feed the Ridge model and get beta_pred for 
    # all those points
    X_col = np.column_stack((model_qresc_f, q2, cost, sint, cos2t, sin2t,
                             q_cost, q_sint, q_cos2t, q_sin2t,
                             q2_cost, q2_sint, cos3t, sin3t, q_cos3t,
                             q_sin3t))

    beta_pred = model.predict(X_col)

    beta_pred = beta_pred.reshape(temp_qresc_m.shape)

    return temp_qresc_m, temp_theta_m, beta_pred
    
# ---------------------------------------------------------------------
def find_max1D(fixq_theta_1D):

    index_max = np.argmax(fixq_theta_1D[:,1])
    theta_max = fixq_theta_1D[index_max,0]
    beta_max = fixq_theta_1D[index_max,1]

    return theta_max, beta_max


# ---------------------------------------------------------------------
def find_max2D(beta_pred_2D, qresc_2D, theta_2D):
    '''
    find the maximum beta and corresponding q and theta in the whole 2D space
    '''
    beta_max_index = np.argmax(beta_pred_2D.flatten())
    beta_max = beta_pred_2D.flatten()[beta_max_index]
    q_max = qresc_2D.flatten()[beta_max_index]
    theta_max = theta_2D.flatten()[beta_max_index]

    return q_max, theta_max, beta_max

if __name__ == "__main__":
    None