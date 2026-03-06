import pathlib
import numpy as np

import db
import rescale
import regression
import plot

cwd = pathlib.Path.cwd()
rescale_bool = True


# ----------------------------------------------------------------------------
def main():
    # Read the database
    db_file = cwd / "../data/urea_test2_v03.db"
    Xbeta_set = db.read_db(db_file)

    # original data
    Xbeta_np = np.array(Xbeta_set)
    q = Xbeta_np[:,0]
    theta = Xbeta_np[:,1]
    beta = Xbeta_np[:,2]

    # rescale q / no need to rescale trigonometric funct
    if rescale_bool:
        qresc = rescale.std_scaling(q)
    else:
        qresc = q

    # derived expansion terms
    #const = np.ones(qresc.shape)
    q2resc = np.square(qresc)
    cost = np.cos(theta)
    cos2t = np.cos(2*theta)
    sint = np.sin(theta)
    sin2t = np.sin(2*theta)
    qresc_cost = qresc*cost
    qresc_cos2t = qresc*cos2t
    qresc_sint = qresc*sint
    qresc_sin2t = qresc*sin2t

    X_col = np.column_stack((qresc, q2resc, cost, sint, cos2t, sin2t,
                             qresc_cost, qresc_sint, qresc_cos2t, qresc_sin2t))
    
    # debug
    #print (X_col)

    R2, coeff, intercept, mse, beta_pred, model = regression.harmt_polq(X_col, beta, alpha = 0)

    exp_data = np.column_stack((q, theta, beta)) # using un-scaled q
    plot.P_2D(exp_data, model, 0.05)

    print("R2:", R2)
    print("mse:", mse)
    print("Coefficients:", coeff)
    print("beta_pred", beta_pred)


if __name__ == "__main__":
    main()